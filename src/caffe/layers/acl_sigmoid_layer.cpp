#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLSigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SigmoidLayer<Dtype>::LayerSetUp(bottom, top);
  ACLBaseActivationLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_SIGMOID;
}

template <typename Dtype>
void ACLSigmoidLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,ActivationLayerInfo::ActivationFunction type){
    ACLBaseActivationLayer<Dtype>::SetupACLLayer(bottom, top,ActivationLayerInfo::ActivationFunction::LOGISTIC);
}
template <typename Dtype>
void ACLSigmoidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SigmoidLayer<Dtype>::Reshape(bottom, top);
  ACLBaseActivationLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_SIGMOID_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        SigmoidLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_cpu(bottom,top);
}

template <typename Dtype>
void ACLSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_SIGMOID_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        SigmoidLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_gpu(bottom,top);
}

template <typename Dtype>
ACLSigmoidLayer<Dtype>::~ACLSigmoidLayer() {
}

INSTANTIATE_CLASS(ACLSigmoidLayer);

}  // namespace caffe
#endif // USE_ACL
