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
void ACLBaseActivationLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,ActivationLayerInfo::ActivationFunction type){

    const unsigned int count  = bottom[0]->count();
    const unsigned int count_ = top[0]->count();
    TensorShape input_shape(count);
    TensorShape output_shape(count_);
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
    ActivationLayerInfo act_info(type);
     
    if(type== ActivationLayerInfo::ActivationFunction::TANH)
      act_info=ActivationLayerInfo(type,1.0,1.0);

   

    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        new_tensor(this->gpu().input,input_shape,(void*)bottom_data);
        new_tensor(this->gpu().output,output_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().output,act_info);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        new_tensor(this->cpu().input,input_shape,(void*)bottom_data);
        new_tensor(this->cpu().output,output_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().output,act_info);
    }
}
template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(Caffe::arm_gpu_mode()){
        Forward_gpu(bottom, top);
        return;
    }        
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    SetupACLLayer(bottom,top);
    tensor_mem(this->cpu().input,(void*)(bottom_data));
    cpu_run();
    tensor_mem((void*)(top_data),this->cpu().output);
}

template <typename Dtype>
void ACLBaseActivationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    SetupACLLayer(bottom,top);
    tensor_mem(this->gpu().input,(void*)(bottom_data));
    gpu_run();
    tensor_mem((void*)(top_data),this->gpu().output);
}

template <typename Dtype>
ACLBaseActivationLayer<Dtype>::~ACLBaseActivationLayer() {
}

INSTANTIATE_CLASS(ACLBaseActivationLayer);

}  // namespace caffe
#endif  // USE_ACL
