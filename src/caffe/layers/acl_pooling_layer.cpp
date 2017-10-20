#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_POOLING;
}
template <typename Dtype>
void ACLPoolingLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    TensorShape in_shape ((unsigned int)this->width_, (unsigned int)this->height_,(unsigned int)this->channels_);
    TensorShape out_shape((unsigned int)this->pooled_width_, (unsigned int)this->pooled_height_,(unsigned int)this->channels_);
    checkreshape(in_shape,Caffe::arm_gpu_mode());
    if (!this->init_layer_) return;
    this->init_layer_=false;
    // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }

    this->force_bypass_acl_path_=false;
    PoolingLayerInfo *pool_info;
    if(this->layer_param_.pooling_param().pool()==PoolingParameter_PoolMethod_MAX)
       pool_info=new PoolingLayerInfo(PoolingType::MAX, this->kernel_w_, PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,DimensionRoundingType::CEIL));
    else
       pool_info=new PoolingLayerInfo(PoolingType::AVG, this->kernel_w_, PadStrideInfo(this->stride_w_,this->stride_h_,this->pad_w_,this->pad_h_,DimensionRoundingType::CEIL));

    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        new_tensor(this->gpu().input,in_shape,(void*)bottom_data);
        new_tensor(this->gpu().output,out_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().output,*pool_info);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        new_tensor(this->cpu().input,in_shape,(void*)bottom_data);
        new_tensor(this->cpu().output,out_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().output,*pool_info);
    }
    delete pool_info;
}
template <typename Dtype>
void ACLPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);

}

template <typename Dtype>
void ACLPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(Caffe::arm_gpu_mode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_POOLING_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_|| this->layer_param_.pooling_param().global_pooling()) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (this->layer_param_.pooling_param().pool()!=PoolingParameter_PoolMethod_MAX && 
      this->layer_param_.pooling_param().pool()!=PoolingParameter_PoolMethod_AVE) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  if (this->kernel_h_!=this->kernel_w_ || top.size()>1) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  if (this->kernel_h_!=2 && this->kernel_h_!=3) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  SetupACLLayer(bottom,top);
  for (int n = 0; n < bottom[0]->num(); ++n) {
        tensor_mem(this->cpu().input,(void*)(bottom_data));
        cpu_run();
        tensor_mem((void*)(top_data),this->cpu().output);
        bottom_data += bottom[0]->offset(1);
        top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
void ACLPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_POOLING_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_|| this->layer_param_.pooling_param().global_pooling()) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (this->layer_param_.pooling_param().pool()!=PoolingParameter_PoolMethod_MAX && 
      this->layer_param_.pooling_param().pool()!=PoolingParameter_PoolMethod_AVE) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  if (this->kernel_h_!=this->kernel_w_) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  if (this->kernel_h_!=2 && this->kernel_h_!=3) {
      PoolingLayer<Dtype>::Forward_cpu(bottom,top);
      return ;
  }
  SetupACLLayer(bottom,top);
  for (int n = 0; n < bottom[0]->num(); ++n) {
        tensor_mem(this->gpu().input,(void*)(bottom_data));
        gpu_run();
        tensor_mem((void*)(top_data),this->gpu().output);
        bottom_data += bottom[0]->offset(1);
        top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
ACLPoolingLayer<Dtype>::~ACLPoolingLayer() {
}

INSTANTIATE_CLASS(ACLPoolingLayer);

}   // namespace caffe
#endif  // USE_ACL
