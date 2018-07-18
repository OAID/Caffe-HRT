#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_PROFILING

#include <iostream>

#include <time.h>

#define REPEAT_TEST

unsigned long get_cur_time(void)
{
   struct timespec tm;

   clock_gettime(CLOCK_MONOTONIC_COARSE, &tm);

   return (tm.tv_sec*1000+tm.tv_nsec/1000000);
}

#endif //USE_PROFILING

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

#ifdef USE_PROFILING

#ifdef LAYER_PERF_STAT
  void  dump_perf_stat(void);
  void  dump_single_layer_io(int idx, Layer<float> * p_layer);
  void  dump_single_layer_perf(int idx, Layer<float> * p_layer,uint64_t total_net_time);
#ifdef REPEAT_TEST
  void collect_layer_stat(vector<vector<perf_stat> * > & all_stat);
  void dump_all_stat(vector <vector<perf_stat>*>& all_stat);
  void reset_layer_stat();
#endif
#endif

#endif //USE_PROFILING

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

#ifdef USE_ACL
  AclEnableSchedule();
#endif
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

#ifdef USE_PROFILING
  unsigned long tstart=get_cur_time();
#endif //USE_PROFILING

  net_->Forward();

#ifdef USE_PROFILING

  unsigned long tend=get_cur_time();

  std::cout<<"used time: "<<tend-tstart<<std::endl;

#ifdef LAYER_PERF_STAT
  dump_perf_stat(); 
#ifdef REPEAT_TEST

   reset_layer_stat();

   vector<vector<perf_stat>* >  all_stat;
   int rep_number=10;

   for(int i=0;i<rep_number;i++)
   {
      net_->Forward();
      collect_layer_stat(all_stat);
      reset_layer_stat();
   }

   //dump stats
   dump_all_stat(all_stat);

   for(int i=0;i<all_stat.size();i++)
         delete all_stat[i];
   
#endif //REPEAT_TEST
#endif //LAYER_PERF_STAT
#endif //USE_PROFILING

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

#ifdef USE_PROFILING

#ifdef LAYER_PERF_STAT

#ifdef REPEAT_TEST
void Classifier::collect_layer_stat(vector<vector<perf_stat>*>& all_stat)
{
   vector<perf_stat > * p_stat;
   perf_stat * p_time_stat;
   const vector<shared_ptr<Layer<float> > >& layers=net_->layers();

   
   p_stat=new vector<perf_stat>;

   for (int i =0;i< layers.size(); i++) {
        p_time_stat=layers[i]->get_time_stat();
        p_stat->push_back(*p_time_stat);

   }

   all_stat.push_back(p_stat);
}

void Classifier::reset_layer_stat(void)
{
   const vector<shared_ptr<Layer<float> > >& layers=net_->layers();
   perf_stat * p_time_stat;

   for (int i =0;i< layers.size(); i++) {
        p_time_stat=layers[i]->get_time_stat();

        p_time_stat->count=0;
        p_time_stat->total=0;
        p_time_stat->used=p_time_stat->start=p_time_stat->end=0;
   }
}

void Classifier::dump_all_stat(vector<vector<perf_stat>*>& all_stat)
{

   struct new_perf_stat {
        perf_stat stat;
        int       idx;
   };
    
   vector<new_perf_stat > layer_stat;
   perf_stat * p_stat;

   uint64_t total_time=0;

   layer_stat.resize(all_stat[0]->size());

   for(int i=0;i<all_stat.size();i++)
   {
      for(int j=0;j<layer_stat.size();j++)
       {
          p_stat=&layer_stat[j].stat;

          p_stat->total+=(*all_stat[i])[j].total;
          p_stat->count+=(*all_stat[i])[j].count;
          total_time+=(*all_stat[i])[j].total;
       }
   }

   total_time=total_time/all_stat.size();

   std::cout<<std::endl<<"----------------------------------"<<std::endl;
   std::cout<<"STATS for "<<all_stat.size()<<" reptitions: ..."<<std::endl;
   std::cout<<"Total time: "<<total_time<<" per forward"<<std::endl;
   std::cout<<"Each layer stats: ..."<<std::endl;


   for(int i=layer_stat.size()-1;i>=0;i--)
   {
      p_stat=&layer_stat[i].stat;

      layer_stat[i].idx=i;

     std::cout<<"  "<<i<<": used time: "<<p_stat->total/all_stat.size();
     std::cout<<" ratio: "<<((float)p_stat->total)/all_stat.size()/total_time*100;
     std::cout<<" enter count: "<<p_stat->count/all_stat.size()<<std::endl;
   }

   std::cout<<std::endl;

   std::cout<<"time cost top 10 layers are: ..."<<std::endl;

   std::sort(layer_stat.begin(),layer_stat.end(),[](const new_perf_stat& a, const new_perf_stat& b)
       {
          if(a.stat.total>b.stat.total)
            return true;
          else
            return false;
       });

   uint64_t  top_total_time=0;

   for(int i=0; i<10; i++)
   {
      p_stat=&layer_stat[i].stat;

     std::cout<<"  "<<layer_stat[i].idx<<": used time: "<<p_stat->total/all_stat.size();
     std::cout<<" ratio: "<<((float)p_stat->total)/all_stat.size()/total_time*100;
     std::cout<<" enter count: "<<p_stat->count/all_stat.size()<<std::endl;
     top_total_time+=p_stat->total;
   }

   std::cout<<"Top cost layers occupied: "<<(float)top_total_time/all_stat.size()/total_time*100<<std::endl;

   std::cout<<std::endl;
}

#endif

void Classifier::dump_single_layer_io(int idx, Layer<float> * p_layer)
{
   const LayerParameter& layer_param=p_layer->layer_param();

   std::cout<<std::endl<<"LAYER IDX: "<<idx<<" name: "<<layer_param.name();
   std::cout<<" type: "<<layer_param.type()<<std::endl;

   const vector<Blob<float>*> *p_bottom_vec=p_layer->saved_bottom;

   for(int i=0;i<layer_param.bottom_size(); i++)
   {
      std::cout<<"bottom "<<layer_param.bottom(i)<<": ";

      Blob<float> * p_blob=(*p_bottom_vec)[i];

      for(int j=0;j<p_blob->num_axes();j++)
      {
          std::cout<<p_blob->shape(j)<<" ";
      }
      std::cout<<std::endl;
   }

   const vector<Blob<float>*> *p_top_vec=p_layer->saved_top;
   for(int i=0;i<layer_param.top_size(); i++)
   {
      std::cout<<"top "<<layer_param.top(i)<<": ";
      Blob<float> * p_blob=(*p_top_vec)[i];

      for(int j=0;j<p_blob->num_axes();j++)
      {
          std::cout<<p_blob->shape(j)<<" ";
      }
      std::cout<<std::endl;
   }
}

void Classifier::dump_single_layer_perf(int idx, Layer<float> * p_layer, uint64_t total_net_time)
{
   const LayerParameter& layer_param=p_layer->layer_param();
   perf_stat * p_time_stat;

   p_time_stat=p_layer->get_time_stat();

   std::cout<<std::endl<<"LAYER IDX: "<<idx<<" name: "<<layer_param.name();
   std::cout<<" type: "<<layer_param.type();
   std::cout<<"  ratio: "<<(float)p_time_stat->total/total_net_time*100<<std::endl;


   std::cout<<"time stat:  total: "<<p_time_stat->total<<" count: "<<p_time_stat->count;
   if(p_time_stat->count)
    {
       std::cout<<" average: "<<((float)p_time_stat->total)/p_time_stat->count;
    }

   std::cout<<" start: "<<p_time_stat->start<<" end: "<<p_time_stat->end;
   std::cout<<std::endl;


} 

void Classifier::dump_perf_stat(void)
{
   uint64_t total_net_time=0;

   const vector<shared_ptr<Layer<float> > >& layers=net_->layers();

   std::cout<<"Input/output shape for each layer ... total: "<<layers.size()<<std::endl;

   for (int i = layers.size() - 1; i >= 0; --i) {
     dump_single_layer_io(i,layers[i].get());
   }


   for (int i = layers.size() - 1; i >= 0; --i) {

     perf_stat * p_time_stat;

     p_time_stat=layers[i]->get_time_stat();

     total_net_time+=p_time_stat->total;

   }
   
   std::cout<<"Time for each layer ... sum of all layers is : ";
   std::cout<<total_net_time<<std::endl;

   for (int i = layers.size() - 1; i >= 0; --i) {

     dump_single_layer_perf(i,layers[i].get(),total_net_time);
   }

}

#endif

#endif //USE_PROFILING

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file = argv[5];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
