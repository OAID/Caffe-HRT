#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_batch_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_BN;
}
template <typename Dtype>
void ACLBatchNormLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    if (!this->init_layer_) return;
    this->init_layer_=false;
    // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }

    this->force_bypass_acl_path_=false;

    TensorShape in_shape ((unsigned int)bottom[0]->width(), (unsigned int)bottom[0]->height(),(unsigned int)bottom[0]->channels(),(unsigned int)bottom[0]->num());
    TensorShape out_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),(unsigned int)top[0]->channels(),(unsigned int)top[0]->num());
    TensorShape mean_shape((unsigned int)this->channels_);
    TensorShape var_shape=mean_shape;
    TensorShape beta_shape=mean_shape;
    TensorShape gamma_shape=mean_shape;
    Dtype beta_val[beta_shape.total_size()];
    Dtype gamma_val[gamma_shape.total_size()];


    for (int i=0;i<beta_shape.total_size();++i) {
        beta_val[i]=0.0;
    }
    for (int i=0;i<gamma_shape.total_size();++i) {
        gamma_val[i]=1.0;
    }
    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        const Dtype* bottom_data = bottom[0]->gpu_data();
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_cpu_scale(this->variance_.count(), scale_factor,
            this->blobs_[0]->gpu_data(), this->mean_.mutable_gpu_data());
        caffe_cpu_scale(this->variance_.count(), scale_factor,
            this->blobs_[1]->gpu_data(), this->variance_.mutable_gpu_data());
        new_tensor(this->gpu().input,in_shape,(void*)bottom_data);
        new_tensor(this->gpu().output,out_shape,(void*)top_data);
        new_tensor(this->gpu().mean,mean_shape);
        new_tensor(this->gpu().var,var_shape);
        new_tensor(this->gpu().beta,beta_shape);
        new_tensor(this->gpu().gamma,gamma_shape);
        tensor_mem(this->gpu().mean,(void*)this->mean_.mutable_gpu_data());
        tensor_mem(this->gpu().var,(void*)this->variance_.mutable_gpu_data());
        tensor_mem(this->gpu().beta,(void*)beta_val);
        tensor_mem(this->gpu().gamma,(void*)gamma_val);
        this->gpu().mean->commit();
        this->gpu().var->commit();
        this->gpu().beta->commit();
        this->gpu().gamma->commit();

#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(this->gpu().input,this->gpu().output,this->gpu().mean,this->gpu().var,this->gpu().beta,this->gpu().gamma,this->eps_);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_cpu_scale(this->variance_.count(), scale_factor,
            this->blobs_[0]->cpu_data(), this->mean_.mutable_cpu_data());
        caffe_cpu_scale(this->variance_.count(), scale_factor,
            this->blobs_[1]->cpu_data(), this->variance_.mutable_cpu_data());
        new_tensor(this->cpu().input,in_shape,(void*)bottom_data);
        new_tensor(this->cpu().output,out_shape,(void*)top_data);
        new_tensor(this->cpu().mean,mean_shape);
        new_tensor(this->cpu().var,var_shape);
        new_tensor(this->cpu().beta,beta_shape);
        new_tensor(this->cpu().gamma,gamma_shape);
        tensor_mem(this->cpu().mean,(void*)this->mean_.mutable_cpu_data());
        tensor_mem(this->cpu().var,(void*)this->variance_.mutable_cpu_data());
        tensor_mem(this->cpu().beta,(void*)beta_val);
        tensor_mem(this->cpu().gamma,(void*)gamma_val);
        this->cpu().mean->commit();
        this->cpu().var->commit();
        this->cpu().beta->commit();
        this->cpu().gamma->commit();

#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(this->cpu().input,this->cpu().output,this->cpu().mean,this->cpu().var,this->cpu().beta,this->cpu().gamma,this->eps_);
    }
}
template <typename Dtype>
void ACLBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::Reshape(bottom, top);

}

template <typename Dtype>
void ACLBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(Caffe::arm_gpu_mode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_BN_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_||!this->use_global_stats_) {
        BatchNormLayer<Dtype>::Forward_cpu(bottom,top);
        return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLLayer(bottom,top);
  tensor_mem(this->cpu().input,(void*)(bottom_data));
  cpu_run();
  tensor_mem((void*)(top_data),this->cpu().output);
}

template <typename Dtype>
void ACLBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if (this->force_bypass_acl_path_||!this->use_global_stats_) {
          BatchNormLayer<Dtype>::Forward_cpu(bottom,top);
          return;
    }
#ifdef USE_PROFILING
  logtime_util log_time(ACL_BN_INFO);
#endif //USE_PROFILING
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLLayer(bottom,top);
  tensor_mem(this->gpu().input,(void*)(bottom_data));
  gpu_run();
  tensor_mem((void*)(top_data),this->gpu().output);
}

template <typename Dtype>
ACLBatchNormLayer<Dtype>::~ACLBatchNormLayer() {
}

INSTANTIATE_CLASS(ACLBatchNormLayer);

}   // namespace caffe
#endif  // USE_ACL
