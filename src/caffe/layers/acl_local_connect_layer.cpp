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
void ACLLocalConnectLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    TensorShape input_shape((unsigned int)bottom[0]->width(), (unsigned int)bottom[0]->height(),(unsigned int)bottom[0]->channels(),(unsigned int)bottom[0]->num());
    checkreshape(input_shape,Caffe::arm_gpu_mode());
    if (!this->init_layer_) return;
    this->init_layer_=false;
  // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }
    this->force_bypass_acl_path_=false;
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    int stride_x =this->stride_;
    int stride_y =this->stride_;
    int pad_x=this->pad_;
    int pad_y=this->pad_;
    unsigned int kernel_x=this->kernel_size_;
    unsigned int kernel_y=this->kernel_size_;
    PadStrideInfo conv_info(stride_x,stride_y,pad_x,pad_y);
    TensorShape weights_shape(kernel_x,kernel_y,(unsigned int)this->channels_, (unsigned int)this->num_output_);
    TensorShape biases_shape ((unsigned int)this->num_output_);
    TensorShape output_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),(unsigned int)top[0]->channels(),(unsigned int)top[0]->num());

    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        //[kernel_x, kernel_y, IFM, OFM]
        new_tensor(this->gpu().weights,weights_shape,(void*)(this->blobs_[0].get()->mutable_gpu_data()));
        tensor_mem(this->gpu().weights,(void*)(this->blobs_[0].get()->mutable_gpu_data()));
        //[OFM]
        if (this->bias_term_) {
            new_tensor(this->gpu().biases,biases_shape,(void*)(this->blobs_[1].get()->mutable_gpu_data()));
            tensor_mem(this->gpu().biases,(void*)(this->blobs_[1].get()->mutable_gpu_data()));
        }

        //[width, height, IFM]
        new_tensor(this->gpu().input,input_shape,(void*)bottom_data);
        //[width, height, OFM]
        new_tensor(this->gpu().output,output_shape,(void*)top_data);
#ifdef USE_PROFILING
        {
            logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().weights,this->gpu().biases,this->gpu().output,conv_info);
#ifdef USE_PROFILING
        }
#endif //USE_PROFILING
#ifdef USE_CONV_CACHE
        for(int i = 0; i < 16; ++i){
            fprintf(stderr, "<GPU>check cache[%d]\n", i);
            if(this->gpu().cache.layer[i] == nullptr){
                this->gpu().cache.layer[i] = this->gpu().layer;
                this->gpu().cache.input[i] = this->gpu().input;
                this->gpu().cache.output[i] = this->gpu().output;
                this->gpu().cache.weights[i] = this->gpu().weights;
                this->gpu().cache.biases[i] = this->gpu().biases;
                break;
            }
        }    
#endif //USE_CONV_CACHE    		
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        //[kernel_x, kernel_y, IFM, OFM]
        new_tensor(this->cpu().weights,weights_shape,(void*)(this->blobs_[0].get()->mutable_cpu_data()));
        tensor_mem(this->cpu().weights,(void*)(this->blobs_[0].get()->mutable_cpu_data()));
        //[OFM]
        if (this->bias_term_) {
            new_tensor(this->cpu().biases,biases_shape,(void*)(this->blobs_[1].get()->mutable_cpu_data()));
            tensor_mem(this->cpu().biases,(void*)(this->blobs_[1].get()->mutable_cpu_data()));
        }

        //[width, height, IFM]
        new_tensor(this->cpu().input,input_shape,(void*)bottom_data);
        //[width, height, OFM]
        new_tensor(this->cpu().output,output_shape,(void*)top_data);
#ifdef USE_PROFILING
        {
            logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().weights,this->cpu().biases,this->cpu().output,conv_info);
#ifdef USE_PROFILING
        }
#endif //USE_PROFILING
    }
}
template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LocalConnectLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if(Caffe::arm_gpu_mode()){
        Forward_gpu(bottom, top);
        return;
    }         
#ifdef USE_PROFILING
    logtime_util log_time(ACL_LC_INFO);
#endif //USE_PROFILING
    if (this->force_bypass_acl_path_) {
        LocalConnectLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }

    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    if (conv_param.kernel_size_size()>2 ) {
        LocalConnectLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    
    SetupACLLayer(bottom,top);
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        tensor_mem(this->cpu().input,(void*)bottom_data);
        cpu_run();
        tensor_mem((void*)top_data,this->cpu().output);
  }
}

template <typename Dtype>
void ACLLocalConnectLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
    logtime_util log_time(ACL_LC_INFO);
#endif //USE_PROFILING
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    if (this->force_bypass_acl_path_) {
        LocalConnectLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    if (conv_param.kernel_size_size()>2 ) {
        LocalConnectLayer<Dtype>::Forward_cpu(bottom,top);
        return;
    }
    SetupACLLayer(bottom,top);
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      tensor_mem(this->gpu().input,(void*)bottom_data);
      gpu_run();
      tensor_mem((void*)top_data,this->gpu().output);
    }
}

template <typename Dtype>
ACLLocalConnectLayer<Dtype>::~ACLLocalConnectLayer() {
}

INSTANTIATE_CLASS(ACLLocalConnectLayer);

}   // namespace caffe
#endif  // USE_ACL
