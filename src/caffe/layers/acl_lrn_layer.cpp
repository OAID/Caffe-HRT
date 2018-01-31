#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_lrn_layer.hpp"

namespace caffe {

const arm_compute::NormType IN_MAP=(arm_compute::NormType)0;
template <typename Dtype>
void ACLLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_LRN;
}
template <typename Dtype>
void ACLLRNLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    arm_compute::TensorShape shape((unsigned int)this->width_,(unsigned int)this->height_, (unsigned int)this->channels_);
    if (is_operator_init_done(shape)) return;
    set_operator_init_done();

    // Initialize ACL.
    arm_compute::NormalizationLayerInfo norm_info(IN_MAP, this->size_, this->alpha_, this->beta_, this->k_);
    if(this->layer_param_.lrn_param().norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL)
       norm_info=arm_compute::NormalizationLayerInfo(IN_MAP, this->size_, this->alpha_, this->beta_, this->k_);
    else
       norm_info=arm_compute::NormalizationLayerInfo(arm_compute::NormType::CROSS_MAP, this->size_, this->alpha_, this->beta_, this->k_);

    new_tensor(input(),shape,InputdataPtr(this,bottom));
    new_tensor(output(),shape,OutputdataPtr(this,top));
    acl_configure(lrn,this,norm_info);

}
template <typename Dtype>
void ACLLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::Reshape(bottom, top);
  return;
}

template <typename Dtype>
bool ACLLRNLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_ || this->layer_param_.lrn_param().norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
        bypass_acl=true;
    }
    return bypass_acl;
}

template <typename Dtype>
void ACLLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(isGPUMode()){
    Forward_gpu(bottom, top);
    return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_LRN_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom, top)) {
      LRNLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLOperator(bottom,top);
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
      for (int n = 0; n < this->num_; ++n) {
          acl_run((void*)(bottom_data+ bottom[0]->offset(n)),(void*)(top_data + top[0]->offset(n)));
      }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
      for (int n = 0; n < bottom[0]->num(); ++n) {
            acl_run((void*)bottom_data,(void*)top_data);
            bottom_data += bottom[0]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
      }
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void ACLLRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_LRN_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom, top)) {
       LRNLayer<Dtype>::Forward_cpu(bottom,top);
       return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLOperator(bottom,top);
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
      for (int n = 0; n < this->num_; ++n) {
          acl_run((void*)(bottom_data+ bottom[0]->offset(n)),(void*)(top_data + top[0]->offset(n)));
      }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
      for (int n = 0; n < bottom[0]->num(); ++n) {
            acl_run((void*)bottom_data,(void*)top_data);
            bottom_data += bottom[0]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
      }
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
ACLLRNLayer<Dtype>::~ACLLRNLayer() {
}

INSTANTIATE_CLASS(ACLLRNLayer);

}   // namespace caffe
#endif  // USE_ACL
