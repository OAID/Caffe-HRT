#ifdef USE_ACL
#include <algorithm>
#include <vector>

#include "caffe/layers/acl_bnll_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLBNLLLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BNLLLayer<Dtype>::LayerSetUp(bottom, top);
  ACLBaseActivationLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_BNLL;
}
template <typename Dtype>
void ACLBNLLLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, ActivationLayerInfo::ActivationFunction type){
    ACLBaseActivationLayer<Dtype>::SetupACLLayer(bottom, top,ActivationLayerInfo::ActivationFunction::SOFT_RELU);
}
template <typename Dtype>
void ACLBNLLLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BNLLLayer<Dtype>::Reshape(bottom, top);
  ACLBaseActivationLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLBNLLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_BNLL_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        BNLLLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_cpu(bottom,top);
}

template <typename Dtype>
void ACLBNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_BNLL_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        BNLLLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_gpu(bottom,top);
}

template <typename Dtype>
ACLBNLLLayer<Dtype>::~ACLBNLLLayer() {
}

INSTANTIATE_CLASS(ACLBNLLLayer);

}  // namespace caffe
#endif  // USE_ACL
