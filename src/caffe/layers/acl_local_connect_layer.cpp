#ifdef USE_ACL
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/acl_local_connect_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LocalConnectLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_LC;
}

template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    arm_compute::TensorShape input_shape((unsigned int)bottom[0]->width(), (unsigned int)bottom[0]->height(),(unsigned int)bottom[0]->channels(),(unsigned int)bottom[0]->num());
    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();

    // Initialize ACL.
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    int stride_x =this->stride_;
    int stride_y =this->stride_;
    int pad_x=this->pad_;
    int pad_y=this->pad_;
    unsigned int kernel_x=this->kernel_size_;
    unsigned int kernel_y=this->kernel_size_;
    arm_compute::PadStrideInfo conv_info(stride_x,stride_y,pad_x,pad_y);
    arm_compute::TensorShape weights_shape(kernel_x,kernel_y,(unsigned int)this->channels_, (unsigned int)this->num_output_);
    arm_compute::TensorShape biases_shape ((unsigned int)this->num_output_);
    arm_compute::TensorShape output_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),(unsigned int)top[0]->channels(),(unsigned int)top[0]->num());

    //[kernel_x, kernel_y, IFM, OFM]
    new_tensor(weights(),weights_shape,GetDataPtr(this,this->blobs_[0].get()));
    //[OFM]
    if (this->bias_term_) {
        new_tensor(biases(),biases_shape,GetDataPtr(this,this->blobs_[1].get()));
    }

    //[width, height, IFM]
    new_tensor(input(),input_shape,InputdataPtr(this,bottom));
    //[width, height, OFM]
    new_tensor(output(),output_shape,OutputdataPtr(this,top));
    acl_configure(lc,this,conv_info);
}
template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LocalConnectLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
bool ACLLocalConnectLayer<Dtype>::Bypass_acl(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_) {
        bypass_acl=true;
    }

    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    if (conv_param.kernel_size_size()>2 ) {
        bypass_acl=true;
    }
    return bypass_acl;
}

template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if(isGPUMode()){
        Forward_gpu(bottom, top);
        return;
    }         
#ifdef USE_PROFILING
    logtime_util log_time(ACL_LC_INFO);
#endif //USE_PROFILING
    if (Bypass_acl(bottom,top)) {
        LocalConnectLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    
    SetupACLOperator(bottom,top);
    caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_LC_INFO);
#endif //USE_PROFILING
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    if (Bypass_acl(bottom,top)) {
        LocalConnectLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    SetupACLOperator(bottom,top);
    caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
ACLLocalConnectLayer<Dtype>::~ACLLocalConnectLayer() {
}

INSTANTIATE_CLASS(ACLLocalConnectLayer);

}   // namespace caffe
#endif  // USE_ACL
