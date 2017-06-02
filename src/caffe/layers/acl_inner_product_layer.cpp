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
void ACLInnerProductLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    TensorShape weights_shape_t((unsigned int)this->K_, (unsigned int)this->N_);
    TensorShape weights_shape((unsigned int)this->N_, (unsigned int)this->K_);
    TensorShape biases_shape((unsigned int)this->N_);
    TensorShape input_shape((unsigned int)this->K_, (unsigned int)this->M_);
    TensorShape output_shape((unsigned int)this->N_, (unsigned int)this->M_);
    checkreshape(input_shape,Caffe::arm_gpu_mode());
    if (!this->init_layer_) return;
    this->init_layer_=false;
    // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }

    bool transpose = !this->layer_param_.inner_product_param().transpose();
    this->force_bypass_acl_path_ = false; 
    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        if (transpose) {
            this->gpu().weights=new_tensor<GPUTensor>(weights_shape_t,(void*)(this->blobs_[0].get()->mutable_gpu_data()));
        }else{
            this->gpu().weights=new_tensor<GPUTensor>(weights_shape,(void*)(this->blobs_[0].get()->mutable_gpu_data()));
        }
        tensor_mem(this->gpu().weights,(void*)(this->blobs_[0].get()->mutable_gpu_data()));
        if (this->bias_term_) {
            this->gpu().biases=new_tensor<GPUTensor>(biases_shape,(void*)(this->blobs_[1].get()->mutable_gpu_data()));
            tensor_mem(this->gpu().biases,(void*)(this->blobs_[1].get()->mutable_gpu_data()));
        }
        this->gpu().input=new_tensor<GPUTensor>(input_shape,(void*)bottom_data);
        this->gpu().output=new_tensor<GPUTensor>(output_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().weights,this->gpu().biases,this->gpu().output,transpose);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        if (transpose) {
            this->cpu().weights=new_tensor<CPUTensor>(weights_shape_t,(void*)(this->blobs_[0].get()->mutable_cpu_data()));
        }else{
            this->cpu().weights=new_tensor<CPUTensor>(weights_shape,(void*)(this->blobs_[0].get()->mutable_cpu_data()));
        }
        tensor_mem(this->cpu().weights,(void*)(this->blobs_[0].get()->mutable_cpu_data()));
        if (this->bias_term_) {
            this->cpu().biases=new_tensor<CPUTensor>(biases_shape,(void*)(this->blobs_[1].get()->mutable_cpu_data()));
            tensor_mem(this->cpu().biases,(void*)(this->blobs_[1].get()->mutable_cpu_data()));
        }
        this->cpu().input=new_tensor<CPUTensor>(input_shape,(void*)bottom_data);
        this->cpu().output=new_tensor<CPUTensor>(output_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().weights,this->cpu().biases,this->cpu().output,transpose);
    }
}
template <typename Dtype>
void ACLInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ACLInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(Caffe::arm_gpu_mode()){
   	Forward_gpu(bottom, top);
   	return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_FC_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_) {
       InnerProductLayer<Dtype>::Forward_cpu(bottom,top);
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
void ACLInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_FC_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_) {
        InnerProductLayer<Dtype>::Forward_cpu(bottom,top);
        return;
  }
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  SetupACLLayer(bottom,top);
  tensor_mem(this->gpu().input,(void*)(bottom_data));
  gpu_run();
  tensor_mem((void*)(top_data),this->gpu().output);
}

template <typename Dtype>
ACLInnerProductLayer<Dtype>::~ACLInnerProductLayer() {
}

INSTANTIATE_CLASS(ACLInnerProductLayer);

}  // namespace caffe
#endif // USE_ACL
