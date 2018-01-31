#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLTanHLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TanHLayer<Dtype>::LayerSetUp(bottom, top);
  ACLBaseActivationLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_TANH;
}

template <typename Dtype>
void ACLTanHLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, arm_compute::ActivationLayerInfo::ActivationFunction type){
    ACLBaseActivationLayer<Dtype>::SetupACLOperator(bottom, top,arm_compute::ActivationLayerInfo::ActivationFunction::TANH);
}

template <typename Dtype>
void ACLTanHLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TanHLayer<Dtype>::Reshape(bottom, top);
  ACLBaseActivationLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
bool ACLTanHLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_) {
        bypass_acl=true;
    }
    return bypass_acl;
}
template <typename Dtype>
void ACLTanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_TANH_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
       TanHLayer<Dtype>::Forward_cpu(bottom,top);
       return;
  }
  ACLBaseActivationLayer<Dtype>::Forward_cpu(bottom,top);
}

template <typename Dtype>
void ACLTanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_TANH_INFO);
#endif //USE_PROFILING
    if (Bypass_acl(bottom,top)) {
         TanHLayer<Dtype>::Forward_cpu(bottom,top);
         return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_gpu(bottom,top);
}

template <typename Dtype>
ACLTanHLayer<Dtype>::~ACLTanHLayer() {
}

INSTANTIATE_CLASS(ACLTanHLayer);

}  // namespace caffe

#endif  // USE_ACL
