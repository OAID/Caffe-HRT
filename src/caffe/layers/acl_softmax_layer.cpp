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
void ACLSoftmaxLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    unsigned int channels = bottom[0]->shape(this->softmax_axis_); 
    arm_compute::TensorShape shape(channels*this->inner_num_);
    if (is_operator_init_done(shape)) return;
    set_operator_init_done();

    // Initialize ACL.
    new_tensor(input(),shape,InputdataPtr(this,bottom));
    new_tensor(output(),shape,OutputdataPtr(this,top));
    acl_configure(softmax,this,NULL);

}
template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SoftmaxLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
bool ACLSoftmaxLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_ || this->inner_num_>1) {
        bypass_acl=true;
    }
    return bypass_acl;
}

template <typename Dtype>
void ACLSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(isGPUMode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
    logtime_util log_time(ACL_SOFTMAX_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
      SoftmaxLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLOperator(bottom,top);

  int channels = bottom[0]->shape(this->softmax_axis_);

  for (int i = 0; i < this->outer_num_; ++i) {
      acl_run((void*)bottom_data,(void*)top_data);
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
  if (Bypass_acl(bottom,top)) {
        SoftmaxLayer<Dtype>::Forward_cpu(bottom,top);
        return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLOperator(bottom,top);
  for (int i = 0; i < this->outer_num_; ++i) {
      acl_run((void*)bottom_data,(void*)top_data);
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
