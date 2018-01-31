#ifdef USE_ACL
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/acl_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void ACLInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_FC;
}
template <typename Dtype>
void ACLInnerProductLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    arm_compute::TensorShape weights_shape_t((unsigned int)this->K_, (unsigned int)this->N_);
    arm_compute::TensorShape weights_shape((unsigned int)this->N_, (unsigned int)this->K_);
    arm_compute::TensorShape biases_shape((unsigned int)this->N_);
    arm_compute::TensorShape input_shape((unsigned int)this->K_, (unsigned int)this->M_);
    arm_compute::TensorShape output_shape((unsigned int)this->N_, (unsigned int)this->M_);
    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();
    // Initialize ACL.

    bool transpose = !this->layer_param_.inner_product_param().transpose();
    if (transpose) {
        new_tensor(weights(),weights_shape_t,GetDataPtr(this,this->blobs_[0].get()));
    }else{
        new_tensor(weights(),weights_shape,GetDataPtr(this,this->blobs_[0].get()));
    }
    if (this->bias_term_) {
        new_tensor(biases(),biases_shape,GetDataPtr(this,this->blobs_[1].get()));
    }
    new_tensor(input(),input_shape,InputdataPtr(this,bottom));
    new_tensor(output(),output_shape,OutputdataPtr(this,top));
    acl_configure(fc,this,transpose);
}
template <typename Dtype>
void ACLInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
bool ACLInnerProductLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_) {
        bypass_acl=true;
    }
    return bypass_acl;
}

template <typename Dtype>
void ACLInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(isGPUMode()){
   	Forward_gpu(bottom, top);
   	return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_FC_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom, top)) {
       InnerProductLayer<Dtype>::Forward_cpu(bottom,top);
       return;
  }
  SetupACLOperator(bottom,top);

  if (this->M_ != 1 && openailab_intfp != 0){
      InnerProductLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }

  // ACL FP
  if(openailab_intfp == 0){
      caffe::acl_run(this,bottom,top);
  }
  return;
}

template <typename Dtype>
void ACLInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_FC_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom, top)) {
        InnerProductLayer<Dtype>::Forward_cpu(bottom,top);
        return;
  }
  SetupACLOperator(bottom,top);
  caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
ACLInnerProductLayer<Dtype>::~ACLInnerProductLayer() {
}

INSTANTIATE_CLASS(ACLInnerProductLayer);

}  // namespace caffe
#endif // USE_ACL
