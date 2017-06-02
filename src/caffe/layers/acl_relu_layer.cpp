#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLULayer<Dtype>::LayerSetUp(bottom, top);
  ACLBaseActivationLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_RELU;
}
template <typename Dtype>
void ACLReLULayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    ACLBaseActivationLayer<Dtype>::SetupACLLayer(bottom, top,ActivationLayerInfo::ActivationFunction::RELU);
}
template <typename Dtype>
void ACLReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLULayer<Dtype>::Reshape(bottom, top);
  ACLBaseActivationLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_RELU_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_) {
      ReLULayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Dtype>::layer_param_.relu_param().negative_slope() != 0) {
    ReLULayer<Dtype>::Forward_cpu(bottom, top);
    return;
  }
  ACLBaseActivationLayer<Dtype>::Forward_cpu(bottom,top);
}

template <typename Dtype>
void ACLReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_RELU_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_) {
      ReLULayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Dtype>::layer_param_.relu_param().negative_slope() != 0) {
    ReLULayer<Dtype>::Forward_cpu(bottom, top);
	return;
  }
  ACLBaseActivationLayer<Dtype>::Forward_gpu(bottom,top);
}

template <typename Dtype>
ACLReLULayer<Dtype>::~ACLReLULayer() {
}

INSTANTIATE_CLASS(ACLReLULayer);

}  // namespace caffe
#endif // USE_ACL
