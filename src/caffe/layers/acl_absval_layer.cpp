#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ACLAbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AbsValLayer<Dtype>::LayerSetUp(bottom, top);
  ACLBaseActivationLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_ABSVAL;
}

template <typename Dtype>
void ACLAbsValLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,ActivationLayerInfo::ActivationFunction type){
    ACLBaseActivationLayer<Dtype>::SetupACLLayer(bottom, top,ActivationLayerInfo::ActivationFunction::ABS);
}

template <typename Dtype>
void ACLAbsValLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AbsValLayer<Dtype>::Reshape(bottom, top);
  ACLBaseActivationLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLAbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_ABSVAL_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        AbsValLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_cpu(bottom,top);
}

template <typename Dtype>
void ACLAbsValLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_ABSVAL_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        AbsValLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    ACLBaseActivationLayer<Dtype>::Forward_gpu(bottom,top);
}

template <typename Dtype>
ACLAbsValLayer<Dtype>::~ACLAbsValLayer() {
}

INSTANTIATE_CLASS(ACLAbsValLayer);

}  // namespace caffe

#endif  // USE_ACL
