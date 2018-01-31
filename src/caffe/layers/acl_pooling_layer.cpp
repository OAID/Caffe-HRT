#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_POOLING;
}
template <typename Dtype>
void ACLPoolingLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    arm_compute::TensorShape in_shape ((unsigned int)this->width_, (unsigned int)this->height_,(unsigned int)this->channels_);
    arm_compute::TensorShape out_shape((unsigned int)this->pooled_width_, (unsigned int)this->pooled_height_,(unsigned int)this->channels_);
    if (is_operator_init_done(in_shape)) return;
    set_operator_init_done();

    // Initialize ACL.
    arm_compute::PoolingLayerInfo pool_info;
    if(this->layer_param_.pooling_param().pool()==PoolingParameter_PoolMethod_MAX)
       pool_info=arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, this->kernel_w_, arm_compute::PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,arm_compute::DimensionRoundingType::CEIL));
    else
       pool_info=arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, this->kernel_w_, arm_compute::PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,arm_compute::DimensionRoundingType::CEIL));

    new_tensor(input(),in_shape,InputdataPtr(this,bottom));
    new_tensor(output(),out_shape,OutputdataPtr(this,top));
    acl_configure(pooling,this,pool_info);

}
template <typename Dtype>
void ACLPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);

}

template <typename Dtype>
bool ACLPoolingLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_|| this->layer_param_.pooling_param().global_pooling()) {
        bypass_acl=true;
    }
    if (this->layer_param_.pooling_param().pool()!=PoolingParameter_PoolMethod_MAX && 
      this->layer_param_.pooling_param().pool()!=PoolingParameter_PoolMethod_AVE) {
        bypass_acl=true;
  }
  if (this->kernel_h_!=this->kernel_w_) {
        bypass_acl=true;
  }
  if (this->kernel_h_!=2 && this->kernel_h_!=3) {
        bypass_acl=true;
  }
    return bypass_acl;
}

template <typename Dtype>
void ACLPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(isGPUMode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_POOLING_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLOperator(bottom,top);
  for (int n = 0; n < bottom[0]->num(); ++n) {
        acl_run((void*)bottom_data,(void*)top_data);
        bottom_data += bottom[0]->offset(1);
        top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
void ACLPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_POOLING_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLOperator(bottom,top);
  for (int n = 0; n < bottom[0]->num(); ++n) {
        acl_run((void*)bottom_data,(void*)top_data);
        bottom_data += bottom[0]->offset(1);
        top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
ACLPoolingLayer<Dtype>::~ACLPoolingLayer() {
}

INSTANTIATE_CLASS(ACLPoolingLayer);

}   // namespace caffe
#endif  // USE_ACL
