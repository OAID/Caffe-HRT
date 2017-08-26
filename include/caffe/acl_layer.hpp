#ifndef CAFFE_ACL_LAYER_HPP_
#define CAFFE_ACL_LAYER_HPP_

#ifdef USE_ACL
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NELocallyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLLocallyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h"
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateKernel.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenate.h"
#include "arm_compute/core/CL/kernels/CLDepthConcatenateKernel.h"
#include "arm_compute/runtime/CL/functions/CLDepthConcatenate.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
using namespace arm_compute;
#define FLAGS_ENABLE_ACL_ABSVAL    0x00000001
#define FLAGS_ENABLE_ACL_BNLL      0x00000002
#define FLAGS_ENABLE_ACL_CONV      0x00000004
#define FLAGS_ENABLE_ACL_FC        0x00000008
#define FLAGS_ENABLE_ACL_LRN       0x00000010
#define FLAGS_ENABLE_ACL_POOLING   0x00000020
#define FLAGS_ENABLE_ACL_RELU      0x00000040
#define FLAGS_ENABLE_ACL_SIGMOID   0x00000080
#define FLAGS_ENABLE_ACL_SOFTMAX   0x00000100
#define FLAGS_ENABLE_ACL_TANH      0x00000200
#define FLAGS_ENABLE_ACL_LC        0x00000400
#define FLAGS_ENABLE_ACL_BN        0x00000800
#define FLAGS_ENABLE_ACL_CONCAT    0x00001000
extern unsigned int bypass_acl_class_layer;
#endif
#ifdef USE_PROFILING
#include "layer.hpp"

#define MASK_LOG_APP_TIME 0x00000001
#define MASK_LOG_ALLOCATE 0x00000002
#define MASK_LOG_RUN      0x00000004
#define MASK_LOG_CONFIG   0x00000008
#define MASK_LOG_COPY     0x00000010
#define MASK_LOG_ABSVAL   0x00000020
#define MASK_LOG_BNLL     0x00000040
#define MASK_LOG_CONV     0x00000080
#define MASK_LOG_FC       0x00000100
#define MASK_LOG_LRN      0x00000200
#define MASK_LOG_POOLING  0x00000400
#define MASK_LOG_RELU     0x00000800
#define MASK_LOG_SIGMOID  0x00001000
#define MASK_LOG_SOFTMAX  0x00002000
#define MASK_LOG_TANH     0x00004000
#define MASK_LOG_LC       0x00008000
#define MASK_LOG_BN       0x00010000
#define MASK_LOG_CONCAT   0x00020000
#define APP_TIME_INFO     MASK_LOG_APP_TIME,"time:       \t"
#define ACL_ALLOCATE_INFO MASK_LOG_ALLOCATE,"allocate:   \t\t"
#define ACL_RUN_INFO      MASK_LOG_RUN,     "run:        \t\t\t"
#define ACL_CONFIG_INFO   MASK_LOG_CONFIG,  "configure:  \t\t\t\t"
#define ACL_COPY_INFO     MASK_LOG_COPY,    "tensor_copy:\t\t\t\t\t"
#define ACL_ABSVAL_INFO   MASK_LOG_ABSVAL,  "ACL_ABSVAL :\t\t\t\t\t\t"
#define ACL_BNLL_INFO     MASK_LOG_BNLL,    "ACL_BNLL   :\t\t\t\t\t\t\t"
#define ACL_CONV_INFO     MASK_LOG_CONV,    "ACL_CONV   :\t\t\t\t\t\t\t\t"
#define ACL_FC_INFO       MASK_LOG_FC,      "ACL_FC     :\t\t\t\t\t\t\t\t\t"
#define ACL_LRN_INFO      MASK_LOG_LRN,     "ACL_LRN    :\t\t\t\t\t\t\t\t\t\t"
#define ACL_POOLING_INFO  MASK_LOG_POOLING, "ACL_POOLING:\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_RELU_INFO     MASK_LOG_RELU,    "ACL_RELU   :\t\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_SIGMOID_INFO  MASK_LOG_SIGMOID, "ACL_SIGMOID:\t\t\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_SOFTMAX_INFO  MASK_LOG_SOFTMAX, "ACL_SOFTMAX:\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_TANH_INFO     MASK_LOG_TANH,    "ACL_TANH   :\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_LC_INFO       MASK_LOG_LC,      "ACL_LC     :\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_BN_INFO       MASK_LOG_BN,      "ACL_BN     :\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
#define ACL_CONCAT_INFO   MASK_LOG_CONCAT,  "ACL_CONCAT :\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
extern unsigned int acl_log_flags;
#endif //USE_PROFILING
namespace caffe {
#ifdef USE_ACL
enum TensorType{
    tensor_input,
    tensor_output,
    tensor_weights,
    tensor_biases,
    tensor_data,
};
template <typename ACLTensor>
class BaseTensor:public ACLTensor{
public:
    BaseTensor(bool share)
       :share_(share),type_(tensor_input),allocate_(false){
    }
    virtual void bindmem(void *mem,bool share){
        mem_=mem;
        share_=share;
    }
    virtual void settensortype(TensorType type){
        type_=type;
    };
    virtual void map(bool blocking = true){}
    virtual void unmap(){}
    virtual void commit(TensorType type=tensor_data);
    int tensor_copy(void * mem, bool toTensor=true);
protected:
    void* mem_;
    bool share_;
    TensorType type_;
    bool allocate_;
};
class GPUTensor:public BaseTensor<CLTensor>{
public:
    explicit GPUTensor(bool share)
       :BaseTensor(share){}
    virtual void map(bool blocking = true){
        if (!allocate_){
            CLTensor::allocator()->allocate();
            allocate_=true;
        }
        CLTensor::map(blocking);
     }
     virtual void unmap(){
        CLTensor::unmap();
     }
};
class CPUTensor:public BaseTensor<Tensor>{
public:
    explicit CPUTensor(bool share)
        :BaseTensor(share){}
    virtual void map(bool blocking = true){
        if (!allocate_){
            Tensor::allocator()->allocate();
            allocate_=true;
        }
    }
    virtual void unmap(){
    }
};
template <typename ACLLayer,typename ACLTensor>
class ACLXPUBaseLayer{
public:
    virtual void commit(){
        if (input) {
            input->commit(tensor_input);
        }
        if (output){
            output->commit(tensor_output);
        }
        if (weights){
            weights->commit(tensor_weights);
        }
        if (biases){
            biases->commit(tensor_biases);
        }
    }
    virtual void run(bool gpu){
        commit();
#ifdef USE_PROFILING
        logtime_util log_time(ACL_RUN_INFO);
#endif //USE_PROFILING
        layer->run();
        if (gpu) {
            // Make sure all the OpenCL jobs are done executing:
            CLScheduler::get().sync();
        }
    }
    virtual bool reshape(TensorShape &shape,TensorType type);
    explicit ACLXPUBaseLayer(){
        layer=nullptr;
        input=nullptr;
        output=nullptr;
        weights=nullptr;
        biases=nullptr;
        mean=nullptr;
        var=nullptr;
        beta=nullptr;
        gamma=nullptr;
#ifdef USE_CONV_CACHE
        for(int i = 0; i < 16; ++i){
           cache.layer[i] = nullptr;
           cache.input[i] = nullptr;
           cache.output[i] = nullptr;
           cache.weights[i] = nullptr;
           cache.biases[i] = nullptr;
        }
#endif //USE_CONV_CACHE    
    }
    virtual void freelayer(){
#ifndef USE_CONV_CACHE
        if (layer) delete layer;
        if (input) delete input;
        if (output) delete output;
        if (weights) delete weights;
        if (biases) delete biases;
        if (mean) delete mean;
        if (var) delete var;
        if (beta) delete beta;
        if (gamma) delete gamma;
#endif //USE_CONV_CACHE    
        layer=nullptr;
        input=nullptr;
        output=nullptr;
        weights=nullptr;
        biases=nullptr;
        mean=nullptr;
        var=nullptr;
        beta=nullptr; 
        gamma=nullptr;
    }
    virtual ~ACLXPUBaseLayer(){
        freelayer();
    }
    ACLLayer *layer;
    ACLTensor *input;
    ACLTensor *output;
    ACLTensor *weights;
    ACLTensor *biases;
    //for BN
    ACLTensor *mean;
    ACLTensor *var;
    ACLTensor *beta; 
    ACLTensor *gamma;
#ifdef USE_CONV_CACHE
    struct{
        ACLLayer *layer[16];
        ACLTensor *input[16];
        ACLTensor *output[16];
        ACLTensor *weights[16];
        ACLTensor *biases[16];
    }cache;
#endif //USE_CONV_CACHE    
};
template <typename GPULayer, typename CPULayer>
class ACLBaseLayer {
public:
    explicit ACLBaseLayer();
    virtual void gpu_run();
    virtual void cpu_run();
    virtual ~ACLBaseLayer();
    virtual GPULayer * new_gpulayer();
    virtual CPULayer * new_cpulayer();
    ACLXPUBaseLayer<GPULayer,GPUTensor>& gpu(){
        return gpu_;
    }
    ACLXPUBaseLayer<CPULayer,CPUTensor>& cpu(){
        return cpu_;
    }
    bool checkreshape(TensorShape shape,bool gpu=false, TensorType type=tensor_input);
    template <typename ACLTensor> bool tensor_mem(ACLTensor *tensor,void *mem,bool share=false);
    template <typename ACLTensor> bool tensor_mem(void *mem,ACLTensor *tensor,bool share=false);
    template <typename ACLTensor> bool new_tensor(ACLTensor *&tensor,TensorShape shape,void *mem=nullptr,bool share=false);
protected:
    ACLXPUBaseLayer<GPULayer,GPUTensor> gpu_;
    ACLXPUBaseLayer<CPULayer,CPUTensor> cpu_;
    bool init_layer_;
    bool force_bypass_acl_path_;

};

#endif
}
#define INSTANTIATE_ACLBASECLASS(GPULayer,CPULayer) \
  template class ACLBaseLayer<GPULayer,CPULayer>; 

#define INSTANTIATE_ACLBASE_FUNCTION(GPULayer,CPULayer,ACLTensor) \
    template bool ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(ACLTensor *tensor,void *mem,bool share); \
    template bool ACLBaseLayer<GPULayer,CPULayer>::tensor_mem(void *mem,ACLTensor *tensor,bool share); \
    template bool ACLBaseLayer<GPULayer,CPULayer>::new_tensor(ACLTensor *&tensor,TensorShape shape,void *mem,bool share); \


#endif
