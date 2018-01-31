#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_concat_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ConcatLayer<Dtype>::LayerSetUp(bottom, top);
  this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_CONCAT;
}
template <typename Dtype>
void ACLConcatLayer<Dtype>::SetupACLOperator(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    unsigned int channels=0;
    for (int i = 0; i < bottom.size(); ++i) {
        channels+=bottom[i]->channels();
    }
    arm_compute::TensorShape out_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),channels);

    if (is_operator_init_done(out_shape,tensor_output)) return;
    set_operator_init_done();

    // Initialize ACL.
    std::vector<arm_compute::TensorShape> shapes;
    for (int i = 0; i < bottom.size(); ++i) {
        arm_compute::TensorShape in_shape((unsigned int)bottom[i]->width(), (unsigned int)bottom[i]->height(),(unsigned int)bottom[i]->channels());
        new_tensor(cinput(i),in_shape,InputdataPtr(this,bottom,i));
    }
    new_tensor(output(),out_shape,OutputdataPtr(this,top));
    acl_configure(concat,this,bottom.size());

}
template <typename Dtype>
void ACLConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    ConcatLayer<Dtype>::Reshape(bottom, top);
}
template <typename Dtype>
bool ACLConcatLayer<Dtype>::Bypass_acl(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    bool bypass_acl=false;
    if (this->force_bypass_acl_path_||this->concat_axis_==0) {
        bypass_acl=true;
    }
    if(isScheduleEnable()){
        bypass_acl=true;
     }
    return bypass_acl;

}

template <typename Dtype>
void ACLConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(isGPUMode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_CONCAT_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
      ConcatLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }

  SetupACLOperator(bottom,top);
  caffe::acl_run(this,bottom,top,false);
}

template <typename Dtype>
void ACLConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_CONCAT_INFO);
#endif //USE_PROFILING
  if (Bypass_acl(bottom,top)) {
      ConcatLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  SetupACLOperator(bottom,top);
  caffe::acl_run(this,bottom,top,false);
}

template <typename Dtype>
ACLConcatLayer<Dtype>::~ACLConcatLayer() {
}

INSTANTIATE_CLASS(ACLConcatLayer);

}   // namespace caffe
#endif  // USE_ACL
