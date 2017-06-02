#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <iomanip>

namespace caffe {

template <typename Dtype>
void dump_blob(const Blob<Dtype> * blob, const char * outfile)
{
   std::ofstream os;
   os.open(outfile);

   os<<setiosflags(ios::fixed);

   for(int i=0;i<blob->LegacyShape(0);i++)
   {

     for(int j=0;j<blob->LegacyShape(1);j++)
     {

        for(int k=0;k<blob->LegacyShape(2);k++)
        {
            for(int l=0;l<blob->LegacyShape(3);l++)
            {
                Dtype data=blob->data_at(i,j,k,l);
                os<<std::setprecision(12)<<data<<", ";
            }
            os<<std::endl;
        }
      os<<std::endl;
    }
     os<<std::endl;
   }

   os.close();

}


template <typename Dtype>
void fill_blob_data(Blob<Dtype >* bottom, int fixed, float val)
{
    for(int i=0;i<bottom->num();i++)
      for(int j=0;j<bottom->channels();j++)
        for(int l=0;l<bottom->height();l++)
          for(int k=0;k<bottom->width();k++)
        {
           int offset;
           Dtype * ptr;

            offset=i*bottom->channels()*bottom->height()*bottom->width()+
                    j*bottom->height()*bottom->width()+
                   l*bottom->width()+k;

           ptr=bottom->mutable_cpu_data();

           if(fixed)
              ptr[offset]=val;
           else
              ptr[offset]=offset+100;

        }


}


template <typename Dtype>
void load_blob_data(Blob<Dtype >* bottom, Dtype * p_data)
{
    for(int i=0;i<bottom->num();i++)
      for(int j=0;j<bottom->channels();j++)
        for(int l=0;l<bottom->height();l++)
          for(int k=0;k<bottom->width();k++)
        {
           int offset;
           Dtype * ptr;

            offset=i*bottom->channels()*bottom->height()*bottom->width()+
                    j*bottom->height()*bottom->width()+
                   l*bottom->width()+k;

           ptr=bottom->mutable_cpu_data();

            ptr[offset]=p_data[offset];

        }

}



// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class ConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


typedef ::testing::Types<CPUDevice<float> > float_only;

#define TestDtypesAndDevices float_only
TYPED_TEST_CASE(ConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  layer_param.set_type("Convolution");
  shared_ptr<Layer<Dtype> > layer=
    LayerRegistry<Dtype>::CreateLayer(layer_param);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

float fail3_weight[]={
-0.850632905960, -1.578843951225, -0.890021681786, 
0.971448659897, -0.538104891777, 0.233876436949, 
-1.242745161057, 2.211859703064, 0.525026142597, 

-1.726792931557, -1.194667577744, 1.119420289993, 
-1.539444208145, 1.725312829018, -1.573384165764, 
0.519557833672, 0.376551657915, -0.615215837955, 

0.758795797825, -0.498177528381, 0.254181325436, 
-0.071698464453, -1.192728281021, 0.776199519634, 
1.837580919266, -0.478745609522, -0.804457962513, 


-2.220808744431, -0.892578184605, -1.422935843468, 
-1.707052111626, -1.837757468224, -1.312300324440, 
-1.251585721970, -1.591378808022, -0.577652215958, 

1.727164268494, 0.176050186157, -1.804216146469, 
0.547152698040, -0.024264926091, -2.040683984756, 
-2.159983396530, 1.692966818810, -1.558626413345, 

-1.242013096809, 0.122898645699, -0.146973758936, 
-0.405744194984, -1.716119289398, 1.215066313744, 
1.061164021492, -0.705341339111, -0.245370775461, 


0.781007647514, -0.104610890150, 2.421228170395, 
0.348720043898, 0.289468020201, 1.841132760048, 
-0.835199236870, -0.242239400744, 1.169079542160, 

0.165550187230, -0.418082803488, 0.479667782784, 
-0.241552516818, 0.767971694469, -0.760977804661, 
-2.419095993042, 0.774254024029, 0.541432976723, 

0.855292022228, -0.144438281655, 0.251998007298, 
-0.242634430528, -0.044748753309, -0.321820944548, 
-0.487676948309, -0.761075556278, -0.646164357662
};

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  vector<int> bottom_shape;
  bottom_shape.push_back(1);
  bottom_shape.push_back(3);
  bottom_shape.push_back(5);
  bottom_shape.push_back(5);

  this->blob_bottom_->Reshape(bottom_shape);

   fill_blob_data(this->blob_bottom_,0,1);

  layer_param.set_type("Convolution");

  shared_ptr<Layer<Dtype> > layer=
   LayerRegistry<Dtype>::CreateLayer(layer_param);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

   //fill_blob_data(layer->blobs()[0].get(),1,1);
   load_blob_data(layer->blobs()[0].get(),fail3_weight);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }


   dump_blob(this->blob_bottom_,"bottom.data");
   dump_blob(this->blob_top_,"top.data");
   dump_blob(this->ref_blob_top_.get(),"reftop.data");
   dump_blob(layer->blobs()[0].get(),"weight.data");
   dump_blob(layer->blobs()[1].get(),"bias.data");
}

}
