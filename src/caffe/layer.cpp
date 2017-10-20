#include "caffe/layer.hpp"

#ifdef USE_PROFILING

#ifdef LAYER_PERF_STAT
#include <time.h>

#endif
#endif //USE_PROFILING

namespace caffe {

INSTANTIATE_CLASS(Layer);

#ifdef USE_PROFILING
#ifdef LAYER_PERF_STAT

/* current timestamp in us */
unsigned long get_cur_time(void)
{
   struct timespec tm;

   clock_gettime(CLOCK_MONOTONIC, &tm);

   return (tm.tv_sec*1000000+tm.tv_nsec/1000);
}


// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);

   saved_top=&top;
   saved_bottom=&bottom;
  
   time_stat_.count++;
   time_stat_.start=get_cur_time();

  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
   time_stat_.end=get_cur_time();
   time_stat_.used=time_stat_.end-time_stat_.start;
   time_stat_.total+=time_stat_.used;
  return loss;
}

#endif
#endif //USE_PROFILING

}  // namespace caffe
