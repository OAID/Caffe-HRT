#ifdef USE_ACL
#include <algorithm>
#include <vector>

#include "caffe/layers/acl_base_activation_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}
template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,arm_compute::ActivationLayerInfo::ActivationFunction type){

    const unsigned int count  = bottom[0]->count();
    const unsigned int count_ = top[0]->count();
    arm_compute::TensorShape input_shape(count);
    arm_compute::TensorShape output_shape(count_);
    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();

    // Initialize ACL.
    arm_compute::ActivationLayerInfo act_info(type);
     
    if(type== arm_compute::ActivationLayerInfo::ActivationFunction::TANH)
      act_info=arm_compute::ActivationLayerInfo(type,1.0,1.0);

    new_tensor(input(),input_shape,(void*)InputdataPtr(this,bottom));
    new_tensor(output(),output_shape,(void*)OutputdataPtr(this,top));
    acl_configure(activation,this,act_info);
}
template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(isGPUMode()){
        ACLBaseActivationLayer<Dtype>::Forward_gpu(bottom, top);
        return;
    }        
    SetupACLOperator(bottom,top);
    caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    SetupACLOperator(bottom,top);
    caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
ACLBaseActivationLayer<Dtype>::~ACLBaseActivationLayer() {
}

INSTANTIATE_CLASS(ACLBaseActivationLayer);

}  // namespace caffe
#endif  // USE_ACL
