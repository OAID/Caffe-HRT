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
void ACLAbsValLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,arm_compute::ActivationLayerInfo::ActivationFunction type){
    ACLBaseActivationLayer<Dtype>::SetupACLOperator(bottom, top,arm_compute::ActivationLayerInfo::ActivationFunction::ABS);
}

template <typename Dtype>
void ACLAbsValLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AbsValLayer<Dtype>::Reshape(bottom, top);
  ACLBaseActivationLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
bool ACLAbsValLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_) {
        bypass_acl=true;
    }
    return bypass_acl;
}

template <typename Dtype>
void ACLAbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_ABSVAL_INFO);
#endif //USE_PROFILING
    if (Bypass_acl(bottom,top)) {
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
    if (Bypass_acl(bottom,top)) {
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
