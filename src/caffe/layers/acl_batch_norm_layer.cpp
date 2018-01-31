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
void ACLBatchNormLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    arm_compute::TensorShape in_shape ((unsigned int)bottom[0]->width(), (unsigned int)bottom[0]->height(),(unsigned int)bottom[0]->channels(),(unsigned int)bottom[0]->num());
    if (is_operator_init_done(in_shape)) return;
    set_operator_init_done();

    // Initialize ACL.
    arm_compute::TensorShape out_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),(unsigned int)top[0]->channels(),(unsigned int)top[0]->num());
    arm_compute::TensorShape mean_shape((unsigned int)this->channels_);
    arm_compute::TensorShape var_shape=mean_shape;
    arm_compute::TensorShape beta_shape=mean_shape;
    arm_compute::TensorShape gamma_shape=mean_shape;
    Dtype beta_val[beta_shape.total_size()];
    Dtype gamma_val[gamma_shape.total_size()];

    for (int i=0;i<beta_shape.total_size();++i) {
        beta_val[i]=0.0;
    }
    for (int i=0;i<gamma_shape.total_size();++i) {
        gamma_val[i]=1.0;
    }

    new_tensor(input(),in_shape,InputdataPtr(this,bottom));
    new_tensor(output(),out_shape,OutputdataPtr(this,top));
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), GetDataPtr(this,&this->mean_));
    caffe_cpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), GetDataPtr(this,&this->variance_));

    new_tensor(mean(),mean_shape,GetDataPtr(this,&this->mean_));
    new_tensor(var(),var_shape,GetDataPtr(this,&this->variance_));
    new_tensor(beta(),beta_shape,(void*)beta_val,true);
    new_tensor(gamma(),gamma_shape,(void*)gamma_val,true);
    acl_configure(bn,this,this->eps_);
}
template <typename Dtype>
void ACLBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::Reshape(bottom, top);

}

template <typename Dtype>
bool ACLBatchNormLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_||!this->use_global_stats_) {
        bypass_acl=true;
    }
    if (isScheduleEnable()) {
        bypass_acl=true;
    }
    return bypass_acl;
}

template <typename Dtype>
void ACLBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(isGPUMode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_BN_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
        BatchNormLayer<Dtype>::Forward_cpu(bottom,top);
        return;
  }
  SetupACLOperator(bottom,top);
  caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
void ACLBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_BN_INFO);
#endif //USE_PROFILING
    if (Bypass_acl(bottom,top)) {
          BatchNormLayer<Dtype>::Forward_cpu(bottom,top);
          return;
    }
  SetupACLOperator(bottom,top);
  caffe::acl_run(this,bottom,top);
}

template <typename Dtype>
ACLBatchNormLayer<Dtype>::~ACLBatchNormLayer() {
}

INSTANTIATE_CLASS(ACLBatchNormLayer);

}   // namespace caffe
#endif  // USE_ACL
