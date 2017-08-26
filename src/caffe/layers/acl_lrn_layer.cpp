#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_lrn_layer.hpp"

namespace caffe {

const NormType IN_MAP=(arm_compute::NormType)0;
template <typename Dtype>
void ACLLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_LRN;
}
template <typename Dtype>
void ACLLRNLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    TensorShape shape((unsigned int)this->width_,(unsigned int)this->height_, (unsigned int)this->channels_);
    checkreshape(shape,Caffe::arm_gpu_mode());
    if (!this->init_layer_) return;
    // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }

    //this->force_bypass_acl_path_=false;
    NormalizationLayerInfo *norm_info;
    if(this->layer_param_.lrn_param().norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL)
       norm_info=new NormalizationLayerInfo(IN_MAP, this->size_, this->alpha_, this->beta_, this->k_);
    else
       norm_info=new NormalizationLayerInfo(NormType::CROSS_MAP, this->size_, this->alpha_, this->beta_, this->k_);

    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        new_tensor(this->gpu().input,shape,(void*)bottom_data);
        new_tensor(this->gpu().output,shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().output,*norm_info);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        new_tensor(this->cpu().input,shape,(void*)bottom_data);
        new_tensor(this->cpu().output,shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().output,*norm_info);
    }
    delete norm_info;
}
template <typename Dtype>
void ACLLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::Reshape(bottom, top);
  return;
}

template <typename Dtype>
void ACLLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(Caffe::arm_gpu_mode()){
    Forward_gpu(bottom, top);
    return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_LRN_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_ || this->layer_param_.lrn_param().norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
      LRNLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLLayer(bottom,top);
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
      for (int n = 0; n < this->num_; ++n) {
          tensor_mem(this->cpu().input,(void*)(bottom_data+ bottom[0]->offset(n)));
          cpu_run();
          tensor_mem((void*)(top_data + top[0]->offset(n)),this->cpu().output);
      }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
      for (int n = 0; n < bottom[0]->num(); ++n) {
            tensor_mem(this->cpu().input,(void*)(bottom_data));
            cpu_run();
            tensor_mem((void*)(top_data),this->cpu().output);
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
  if (this->force_bypass_acl_path_) {
       LRNLayer<Dtype>::Forward_cpu(bottom,top);
       return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLLayer(bottom,top);
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
      for (int n = 0; n < this->num_; ++n) {
          tensor_mem(this->gpu().input,(void*)(bottom_data+ bottom[0]->offset(n)));
          gpu_run();
          tensor_mem((void*)(top_data + top[0]->offset(n)),this->gpu().output);
      }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
      for (int n = 0; n < bottom[0]->num(); ++n) {
            tensor_mem(this->gpu().input,(void*)(bottom_data));
            gpu_run();
            tensor_mem((void*)(top_data),this->gpu().output);
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
