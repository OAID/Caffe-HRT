#ifdef USE_ACL
#include "caffe/acl_layer.hpp"

unsigned int bypass_acl_class_layer =    (0 | \
                                          /*0xffffffff |*/ \
                                          /*FLAGS_ENABLE_ACL_FC |*/ \
                                          /*FLAGS_ENABLE_ACL_LRN |*/ \
                                          0 );

#ifdef USE_PROFILING

#include "arm_neon.h"

unsigned int acl_log_flags = (0 | \
                              MASK_LOG_APP_TIME | \
                            /*MASK_LOG_ALLOCATE | */\
                            /*MASK_LOG_ALLOCATE | */\
                            /*MASK_LOG_RUN      | */\
                            /*MASK_LOG_CONFIG   | */\
                            /*MASK_LOG_COPY     | */\
                              MASK_LOG_ABSVAL   | \
                              MASK_LOG_BNLL     | \
                              MASK_LOG_CONV     | \
                              MASK_LOG_FC       | \
                              MASK_LOG_LRN      | \
                              MASK_LOG_POOLING  | \
                              MASK_LOG_RELU     | \
                              MASK_LOG_SIGMOID  | \
                              MASK_LOG_SOFTMAX  | \
                              MASK_LOG_TANH     | \
                              MASK_LOG_LC       | \
                              MASK_LOG_BN       | \
                              MASK_LOG_CONCAT   | \
                              0);                                          
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#endif //USE_PROFILING

namespace caffe {
template <typename GPULayer, typename CPULayer>
ACLBaseLayer<GPULayer,CPULayer>::ACLBaseLayer()
    :init_layer_(true),force_bypass_acl_path_(false){
  const char* pBypassACL;
  pBypassACL = getenv ("BYPASSACL");
  if (pBypassACL){
    unsigned int bacl;
    sscanf(pBypassACL,"%i", &bacl);
	if(bacl != bypass_acl_class_layer){
	    bypass_acl_class_layer = bacl;
        printf("BYPASSACL<%s>\n", pBypassACL);
        printf("BYPASSACL: %x\n", bypass_acl_class_layer);
	}
  }
#ifdef USE_PROFILING
  const char* pLogACL;
  pLogACL    = getenv("LOGACL");
  if (pLogACL){
    unsigned int alf;
    sscanf(pLogACL,"%i", &alf);
	if (alf != acl_log_flags){
	    acl_log_flags = alf;
        printf("LOGACL<%s>\n", pLogACL);
        printf("LOGACL: %x\n", acl_log_flags);
	}
  }
#endif //USE_PROFILING
}
template <typename GPULayer, typename CPULayer>
void ACLBaseLayer<GPULayer,CPULayer>::gpu_run() {
    gpu_.run(true);
}
template <typename GPULayer, typename CPULayer>
void ACLBaseLayer<GPULayer,CPULayer>::cpu_run() {
    cpu_.run(false);
}

template <typename GPULayer, typename CPULayer>
ACLBaseLayer<GPULayer,CPULayer>::~ACLBaseLayer(){
}
template <typename GPULayer, typename CPULayer>
template <typename ACLTensor> bool ACLBaseLayer<GPULayer,CPULayer>::new_tensor(ACLTensor *&tensor,TensorShape shape,void *mem,bool share)
{
    tensor=new ACLTensor(share);
#if 1    //F32
    tensor->allocator()->init(TensorInfo(shape, Format::F32));
#else  //F16
    tensor->allocator()->init(TensorInfo(shape, Format::F16));
#endif    
    tensor->bindmem(mem,share);
    return true;
}

template <typename ACLTensor>
void BaseTensor<ACLTensor>::commit(TensorType type){
    settensortype(type);
    if (!share_&&mem_) {
        if (!allocate_){ 
#ifdef USE_PROFILING
            logtime_util log_time(ACL_ALLOCATE_INFO);
#endif //USE_PROFILING
            ACLTensor::allocator()->allocate(); 
            allocate_=true;
        }
        if (type_!= tensor_output) {
           tensor_copy(mem_);
        }
        mem_=nullptr;
    }
}

template <typename ACLTensor>
int BaseTensor<ACLTensor>::tensor_copy(void * mem,bool toTensor)
{
#ifdef USE_PROFILING
    logtime_util log_time(ACL_COPY_INFO);
#endif //USE_PROFILING
    arm_compute::Window window;
    ACLTensor* tensor=this;
    window.use_tensor_dimensions(tensor->info()->tensor_shape(), /* first_dimension =*/Window::DimY); // Iterate through the rows (not each element)
    int width = tensor->info()->tensor_shape()[0]; //->dimension(0); //window.x().end() - window.x().start(); // + 1;
    int height = tensor->info()->tensor_shape()[1]; //->dimension(1); //window.y().end() - window.y().start(); // + 1;
    int deepth = tensor->info()->tensor_shape()[2];
    map();
    // Create an iterator:
    arm_compute::Iterator it(tensor, window);
    // Except it works for an arbitrary number of dimensions
    if (toTensor) { //mem->tensor
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
        {
#if 0 //F16
            if (tensor->info()->element_size() ==2)
            {
                for(int i = 0; i < width; i+= 4){
                    auto pa = (float32x4_t*)((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width + id.x() + i) * 4);
                    *(float16x4_t*)(((char*)it.ptr()) + i*2) = vcvt_f16_f32(*pa);
                }
            }
            else{
#endif
                memcpy(it.ptr(), ((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width + id.x()) * tensor->info()->element_size()), width * tensor->info()->element_size());
#if 0 //F16
            }
#endif
        },
        it);
    }else{ //tensor-->mem
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
        {
#if 0 //F16		
            if (tensor->info()->element_size() ==2)
            {
                for(int i = 0; i < width; i+= 4){
                    auto pa = (float32x4_t*)(((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width + id.x() + i) * 4));
                    *pa = vcvt_f32_f16(*(float16x4_t*)(((char*)it.ptr()) + i*2));
                }
            }
            else{
#endif			
                memcpy(((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width) * tensor->info()->element_size()), it.ptr(), width * tensor->info()->element_size());
#if 0 //F16				
            }
#endif			
        },
        it);
    }
    unmap();

    return 0;
}

template <typename GPULayer, typename CPULayer>
template <typename ACLTensor> bool  ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(ACLTensor *tensor,void *mem,bool share)
{
    tensor->bindmem(mem,share);
    return true;
}

template <typename GPULayer, typename CPULayer>
template <typename ACLTensor> bool  ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(void *mem,ACLTensor *tensor,bool share)
{
    if (mem==tensor->buffer()) return true;
    if (!share) {
     tensor->tensor_copy(mem,false);
    }
    return true;
}


template <typename GPULayer, typename CPULayer>
bool ACLBaseLayer<GPULayer,CPULayer>::checkreshape(TensorShape shape,bool gpu, TensorType type)
{
    if (gpu) {
        init_layer_ = gpu_.reshape(shape,type);
    }else{
        init_layer_ = cpu_.reshape(shape,type);
    }
    return init_layer_;
}

template <typename GPULayer, typename CPULayer>
GPULayer * ACLBaseLayer<GPULayer,CPULayer>::new_gpulayer(){
        gpu_.layer= new GPULayer;
        return gpu_.layer;
}
template <typename GPULayer, typename CPULayer>
CPULayer * ACLBaseLayer<GPULayer,CPULayer>::new_cpulayer(){
        cpu_.layer= new CPULayer;
        return cpu_.layer;
}
template <typename ACLLayer,typename ACLTensor>
bool ACLXPUBaseLayer<ACLLayer,ACLTensor>::reshape(TensorShape &shape,TensorType type)
{
    TensorShape _shape;
    if (!layer) return true;
#ifdef USE_CONV_CACHE
    if (tensor_input == type){
        _shape = input->info()->tensor_shape();
        if (_shape.total_size()==shape.total_size() && _shape[0]==shape[0] && _shape[1]==shape[1]) {
            return false;
        }
        for(int i = 0; i < 16; ++i){
            if(cache.input[i] == nullptr) break;
            _shape = cache.input[i]->info()->tensor_shape();
            if (_shape.total_size()==shape.total_size() && _shape[0]==shape[0] && _shape[1]==shape[1]) {
                this->layer = cache.layer[i];
                this->input = cache.input[i];
                this->output = cache.output[i];
                this->weights = cache.weights[i];
                this->biases = cache.biases[i]; 
                return false;
            }
        }
    }
#endif //USE_CONV_CACHE    
    switch (type) {
    case tensor_biases:
        _shape = biases->info()->tensor_shape();
        break;
    case tensor_weights:
        _shape = weights->info()->tensor_shape();
        break;
    case tensor_output:
        _shape = output->info()->tensor_shape();
        break;
    case tensor_input:
    default:
        _shape = input->info()->tensor_shape();
        break;
    }
    if (_shape.total_size()==shape.total_size() && _shape[0]==shape[0] && _shape[1]==shape[1]) {
        return false;
    }
    freelayer();
    return true;
}

INSTANTIATE_ACLBASECLASS(CLNormalizationLayer,NENormalizationLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLNormalizationLayer,NENormalizationLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLNormalizationLayer,NENormalizationLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLActivationLayer,NEActivationLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLActivationLayer,NEActivationLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLActivationLayer,NEActivationLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLPoolingLayer,NEPoolingLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLPoolingLayer,NEPoolingLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLPoolingLayer,NEPoolingLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLSoftmaxLayer,NESoftmaxLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLSoftmaxLayer,NESoftmaxLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLSoftmaxLayer,NESoftmaxLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLFullyConnectedLayer,NEFullyConnectedLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLFullyConnectedLayer,NEFullyConnectedLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLFullyConnectedLayer,NEFullyConnectedLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLConvolutionLayer,NEConvolutionLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLConvolutionLayer,NEConvolutionLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLConvolutionLayer,NEConvolutionLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLConvolutionLayer,NEDirectConvolutionLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLConvolutionLayer,NEDirectConvolutionLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLConvolutionLayer,NEDirectConvolutionLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLBatchNormalizationLayer,NEBatchNormalizationLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLBatchNormalizationLayer,NEBatchNormalizationLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLBatchNormalizationLayer,NEBatchNormalizationLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLLocallyConnectedLayer,NELocallyConnectedLayer); 
  INSTANTIATE_ACLBASE_FUNCTION(CLLocallyConnectedLayer,NELocallyConnectedLayer,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLLocallyConnectedLayer,NELocallyConnectedLayer,CPUTensor);
INSTANTIATE_ACLBASECLASS(CLDepthConcatenate,NEDepthConcatenate); 
  INSTANTIATE_ACLBASE_FUNCTION(CLDepthConcatenate,NEDepthConcatenate,GPUTensor);
  INSTANTIATE_ACLBASE_FUNCTION(CLDepthConcatenate,NEDepthConcatenate,CPUTensor);
}

#endif
