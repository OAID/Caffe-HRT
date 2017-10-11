#ifdef USE_ACL
#include <vector>

#include "caffe/layers/acl_concat_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACLConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ConcatLayer<Dtype>::LayerSetUp(bottom, top);
  //this->force_bypass_acl_path_= bypass_acl_class_layer & FLAGS_ENABLE_ACL_CONCAT;
  this->force_bypass_acl_path_= true;
}
template <typename Dtype>
void ACLConcatLayer<Dtype>::SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    unsigned int channels=0;
    for (int i = 0; i < bottom.size(); ++i) {
        channels+=bottom[i]->channels();
    }
    TensorShape out_shape((unsigned int)top[0]->width(), (unsigned int)top[0]->height(),channels);

    if (!this->init_layer_) return;
    this->init_layer_=false;
    // Initialize ACL.
    if (Caffe::arm_gpu_mode()) {
        new_gpulayer();
    }else{
        new_cpulayer();
    }

    this->force_bypass_acl_path_=false;
	
    if (Caffe::arm_gpu_mode()) {
        Dtype *top_data = top[0]->mutable_gpu_data(); 
        for (int i = 0; i < bottom.size(); ++i) {
          const Dtype* bottom_data = bottom[i]->gpu_data();
          TensorShape vec_shape((unsigned int)bottom[i]->width(), (unsigned int)bottom[i]->height(),(unsigned int)bottom[0]->channels());
          GPUTensor *vector;
          new_tensor(vector,vec_shape,(void*)bottom_data);
          tensor_mem(vector,(void*)bottom_data);
          vector->commit();
          gpu_vectors.push_back(vector);
        }
        new_tensor(this->gpu().output,out_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->gpu().layer->configure(gpu_vectors,this->gpu().output);
    }else{
        Dtype *top_data = top[0]->mutable_cpu_data(); 
        for (int i = 0; i < bottom.size(); ++i) {
          const Dtype* bottom_data = bottom[i]->cpu_data();
          TensorShape vec_shape((unsigned int)bottom[i]->width(), (unsigned int)bottom[i]->height(),(unsigned int)bottom[0]->channels());
          CPUTensor *vector;
          new_tensor(vector,vec_shape,(void*)bottom_data);
          tensor_mem(vector,(void*)bottom_data);
          vector->commit();
          cpu_vectors.push_back(vector);
        }
        new_tensor(this->cpu().output,out_shape,(void*)top_data);
#ifdef USE_PROFILING
        logtime_util log_time(ACL_CONFIG_INFO);
#endif //USE_PROFILING
        this->cpu().layer->configure(cpu_vectors,this->cpu().output);
    }
}
template <typename Dtype>
void ACLConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ConcatLayer<Dtype>::Reshape(bottom, top);

}

template <typename Dtype>
void ACLConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(Caffe::arm_gpu_mode()){
      Forward_gpu(bottom, top);
      return;
  }         
#ifdef USE_PROFILING
  logtime_util log_time(ACL_CONCAT_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_||this->concat_axis_==0) {
      ConcatLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }

  Dtype* top_data = top[0]->mutable_cpu_data();
  SetupACLLayer(bottom,top);
  cpu_run();
  tensor_mem((void*)(top_data),this->cpu().output);
}

template <typename Dtype>
void ACLConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#ifdef USE_PROFILING
  logtime_util log_time(ACL_CONCAT_INFO);
#endif //USE_PROFILING
  if (this->force_bypass_acl_path_||this->concat_axis_==0) {
      ConcatLayer<Dtype>::Forward_cpu(bottom,top);
      return;
  }
  Dtype* top_data = top[0]->mutable_gpu_data();
  SetupACLLayer(bottom,top);
  gpu_run();
  tensor_mem((void*)(top_data),this->gpu().output);
}

template <typename Dtype>
ACLConcatLayer<Dtype>::~ACLConcatLayer() {
    if(this->force_bypass_acl_path_)return;
    for (int i =0; i < cpu_vectors.size(); i ++) {
        delete cpu_vectors[i];
    }
    for (int i =0; i < gpu_vectors.size(); i ++) {
        delete gpu_vectors[i];
    }
    cpu_vectors.erase(cpu_vectors.begin());
    gpu_vectors.erase(gpu_vectors.begin());
}

INSTANTIATE_CLASS(ACLConcatLayer);

}   // namespace caffe
#endif  // USE_ACL
