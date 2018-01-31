#ifdef USE_ACL
#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/acl_conv_layer.hpp"

namespace caffe {

bool use_direct_conv_=false;
template <typename Dtype>
void ACLConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_CONV;
}

template <typename Dtype>
void ACLConvolutionLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    arm_compute::TensorShape input_shape((unsigned int)bottom[0]->width(), (unsigned int)bottom[0]->height(),(unsigned int)bottom[0]->channels(),(unsigned int)bottom[0]->num());
    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();

  // Initialize ACL.
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    int stride_x =this->stride_.mutable_cpu_data()[1];
    int stride_y =this->stride_.mutable_cpu_data()[0];
    int pad_x=this->pad_.mutable_cpu_data()[1];
    int pad_y=this->pad_.mutable_cpu_data()[0];
    unsigned int kernel_x=this->kernel_shape_.mutable_cpu_data()[1];
    unsigned int kernel_y=this->kernel_shape_.mutable_cpu_data()[0];
    arm_compute::PadStrideInfo conv_info(stride_x,stride_y,pad_x,pad_y);
    arm_compute::TensorShape weights_shape(kernel_x,kernel_y,(unsigned int)this->channels_/this->group_, (unsigned int)this->num_output_);
    arm_compute::TensorShape biases_shape ((unsigned int)this->num_output_);
    arm_compute::TensorShape output_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),(unsigned int)top[0]->channels(),(unsigned int)top[0]->num());
    group()=this->group_;

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

    acl_configure(conv,this,conv_info);
}
template <typename Dtype>
void ACLConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
bool ACLConvolutionLayer<Dtype>::Bypass_acl(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_|| ((openailab_intfp==0) && (this->group_>=5)) //for performance, more groups impact GPU performance
       || ((openailab_intfp != 0 && (top[0]->channels() / this->group_ == 1)))) {
        bypass_acl=true;
    }

    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    if (conv_param.kernel_size_size()>2 || this->num_spatial_axes_>2 || this->num_spatial_axes_==0) {
        bypass_acl=true;
    }
    /* check dilation */
    int dilated=0;

    for(int i=0;i<this->num_spatial_axes_;i++)
    {
        const int *p=this->dilation_.cpu_data();

        if(p[i]!=1) 
           dilated=1;
    }
    if(dilated) {
        bypass_acl=true;
     }


    if((this->kernel_shape_.mutable_cpu_data()[1]==1||this->kernel_shape_.mutable_cpu_data()[0]==1) &&
        isScheduleEnable()){
        bypass_acl=true;
     }
    if((this->kernel_shape_.mutable_cpu_data()[1]==3||this->kernel_shape_.mutable_cpu_data()[0]==3) &&
        (bottom[0]->channels()<150) && isScheduleEnable()){
        bypass_acl=true;
     }

    return bypass_acl;
}

template <typename Dtype>
void ACLConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if(isGPUMode()){
        Forward_gpu(bottom, top);
        return;
    }         
#ifdef USE_PROFILING
    logtime_util log_time(ACL_CONV_INFO);
#endif //USE_PROFILING

    if (Bypass_acl(bottom,top)) {
        ConvolutionLayer<Dtype>::Forward_cpu(bottom,top);
        return;
     }
   
    SetupACLOperator(bottom,top);

   // acl fp
    if (openailab_intfp==0){
        caffe::acl_run(this,bottom,top);
    }
    return;
}

template <typename Dtype>
void ACLConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_CONV_INFO);
#endif //USE_PROFILING
    if (Bypass_acl(bottom,top)) {
        ConvolutionLayer<Dtype>::Forward_cpu(bottom,top);
        return;
     }
    SetupACLOperator(bottom,top);
    caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
ACLConvolutionLayer<Dtype>::~ACLConvolutionLayer() {
}

#ifdef USE_ACL
INSTANTIATE_CLASS(ACLConvolutionLayer);
#endif

}   // namespace caffe
#endif  // USE_ACL
