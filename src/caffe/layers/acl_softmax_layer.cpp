#ifdef USE_ACL
#include <vector>
#include "caffe/layers/acl_softmax_layer.hpp"
#include <unistd.h>


namespace caffe {

template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SoftmaxLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_SOFTMAX;
}
template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    unsigned int channels = bottom[0]->shape(this->softmax_axis_); 
    TensorShape shape(channels*this->inner_num_);
    checkreshape(shape,Caffe::arm_gpu_mode());
    if (!this->init_layer_) return;
    this->init_layer_=false;

    // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }

    //this->force_bypass_acl_path_=false;
    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        new_tensor(this->gpu().input,shape,(void*)bottom_data);
        new_tensor(this->gpu().output,shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().output);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        new_tensor(this->cpu().input,shape,(void*)bottom_data);
        new_tensor(this->cpu().output,shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().output);
    }
}
template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SoftmaxLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(Caffe::arm_gpu_mode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
    logtime_util log_time(ACL_SOFTMAX_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_ || this->inner_num_>1) {
      SoftmaxLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLLayer(bottom,top);

  int channels = bottom[0]->shape(this->softmax_axis_);

  for (int i = 0; i < this->outer_num_; ++i) {
      tensor_mem(this->cpu().input,(void*)(bottom_data));
      cpu_run();
      tensor_mem((void*)(top_data),this->cpu().output);
      top_data += channels;
      bottom_data += channels;
  }
}

template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_SOFTMAX_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_|| this->inner_num_>1) {
        SoftmaxLayer<Dtype>::Forward_cpu(bottom,top);
        return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLLayer(bottom,top);
  for (int i = 0; i < this->outer_num_; ++i) {
      tensor_mem(this->gpu().input,(void*)(bottom_data));
      gpu_run();
      tensor_mem((void*)(top_data),this->gpu().output);
      top_data += this->inner_num_;
      bottom_data += this->inner_num_;
  }
}

template <typename Dtype>
ACLSoftmaxLayer<Dtype>::~ACLSoftmaxLayer() {
}

INSTANTIATE_CLASS(ACLSoftmaxLayer);
}  // namespace caffe

#endif  // USE_ACL
