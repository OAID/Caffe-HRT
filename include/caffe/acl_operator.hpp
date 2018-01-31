#ifndef CAFFE_ACL_LAYER_HPP_
#define CAFFE_ACL_LAYER_HPP_

#ifdef USE_ACL
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NELocallyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/Tensor.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/functions/CLDepthConcatenateLayer.h"
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLLocallyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "acl_tensor.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
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
extern unsigned int openailab_intfp;
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
    tensor_mean,
    tensor_var,
    tensor_beta,
    tensor_gamma,
    tensor_concat,
    tensor_data,
};
enum OperatorState{
    operator_not_init,
    operator_init_done,
    operator_reinit,
};
enum OperateType{
    operate_type_pooling,
    operate_type_activation,
    operate_type_lrn,
    operate_type_conv,
    operate_type_lc,
    operate_type_fc,
    operate_type_bn,
    operate_type_softmax,
    operate_type_concat,
};
class BaseACLTensor{
public:
    BaseACLTensor()
         :type_(tensor_input),allocate_(false){
    }
    virtual void bindmem(void *mem){
        mem_=mem;
    }
    virtual void settensortype(TensorType type){
        type_=type;
    };
    virtual void map(bool blocking = true){
    }
    virtual void unmap(){}
    virtual void commit(TensorType type=tensor_data){}
    int tensor_copy(arm_compute::ITensor* tensor,void * mem, bool toTensor=true);
protected:
    void* mem_;
    TensorType type_;
    bool allocate_;
};
class ACLTensor:public BaseACLTensor,public Tensor{
public:
    ACLTensor(arm_compute::TensorInfo &&info)
       :Tensor(info){
    }
    virtual void map(bool blocking = true){
        if (!allocate_){
            Tensor::allocate();
            allocate_=true;
        }
        Tensor::map(blocking);
    }
    virtual int tensor_copy(void * mem, bool toTensor=true){
        auto acl_tensor=this;
        arm_compute::ITensor* tensor=acl_tensor->tensor();
        BaseACLTensor::tensor_copy(tensor,mem,toTensor);
        return 0;
    }
    virtual void unmap(){Tensor::unmap();}
    virtual void commit(TensorType type=tensor_data);
};
class ACLSubTensor:public BaseACLTensor,public SubTensor{
public:
    ACLSubTensor(std::unique_ptr<ACLTensor> &parent,arm_compute::TensorShape &shape,arm_compute::Coordinates& coord)
       :SubTensor(parent.get(),shape,coord){
    }
    virtual int tensor_copy(void * mem, bool toTensor=true){
        return 0;
    }
};

template <typename T>
class TensorPair{
public:
    TensorPair(){}
    ~TensorPair(){}
    TensorType type;
    std::unique_ptr<T> tensor;
};
template <typename T>
std::unique_ptr<T> &tensor_item(std::vector<std::unique_ptr<TensorPair<T>>>& pool,TensorType type,int idx){
    int count=0;
    for (auto &item: pool) {
        if(item.get()->type==type){
            ++count;
        }
        if(item.get()->type==type && idx==count-1){
            return item.get()->tensor;
        }
    }
    pool.push_back((std::unique_ptr<TensorPair<T>>)std::move(new TensorPair<T>));
    auto item=pool[pool.size()-1].get();
    item->type=type;
    item->tensor=NULL;
    return item->tensor;
}
class ACLOperator {
public:
    virtual void commit(){
        for (auto & item: tensor_pool_) {
            if(item.get()->tensor)item.get()->tensor->commit(item.get()->type);
        }
    }
    inline void run(){
        commit();
    #ifdef USE_PROFILING
            logtime_util log_time(ACL_RUN_INFO);
    #endif //USE_PROFILING
       for(auto &c : funcs_)
       {
           c->run();
       }
    }

    inline std::vector<std::unique_ptr<arm_compute::IFunction>> &funcs(){return funcs_;}

    inline std::unique_ptr<ACLSubTensor> &sinput(int idx=0){return tensor_item(subtensor_pool_,tensor_input,idx);}
    inline std::unique_ptr<ACLSubTensor> &soutput(int idx=0){return tensor_item(subtensor_pool_,tensor_output,idx);}
    inline std::unique_ptr<ACLSubTensor> &sweights(int idx=0){return tensor_item(subtensor_pool_,tensor_weights,idx);}
    inline std::unique_ptr<ACLSubTensor> &sbiases(int idx=0){return tensor_item(subtensor_pool_,tensor_biases,idx);}

    inline std::unique_ptr<ACLTensor> &cinput(int idx=0){return tensor_item(tensor_pool_,tensor_concat,idx);}
    inline std::unique_ptr<ACLTensor> &input(int idx=0){return tensor_item(tensor_pool_,tensor_input,idx);}
    inline std::unique_ptr<ACLTensor> &output(int idx=0){return tensor_item(tensor_pool_,tensor_output,idx);}
    inline std::unique_ptr<ACLTensor> &weights(int idx=0){return tensor_item(tensor_pool_,tensor_weights,idx);}
    inline std::unique_ptr<ACLTensor> &biases(int idx=0){return tensor_item(tensor_pool_,tensor_biases,idx);}
    inline std::unique_ptr<ACLTensor> &mean(int idx=0){return tensor_item(tensor_pool_,tensor_mean,idx);}
    inline std::unique_ptr<ACLTensor> &var(int idx=0){return tensor_item(tensor_pool_,tensor_var,idx);}
    inline std::unique_ptr<ACLTensor> &beta(int idx=0){return tensor_item(tensor_pool_,tensor_beta,idx);}
    inline std::unique_ptr<ACLTensor> &gamma(int idx=0){return tensor_item(tensor_pool_,tensor_gamma,idx);}
    inline std::unique_ptr<ACLTensor> &tensor(TensorType type){
        switch (type) {
        case tensor_biases:
            return biases();
            break;
        case tensor_weights:
            return weights();
            break;
        case tensor_output:
            return output();
            break;
        default:
        case tensor_input:
            return input();
            break;
        }
        return input();
    }


    explicit ACLOperator(const LayerParameter& param);
    virtual ~ACLOperator();
    inline TargetHint getTargetHint(){
#ifdef USE_OPENCL
        if (target_hint_==TargetHint::DONT_CARE) {
            if (Caffe::arm_gpu_mode()) {
                return TargetHint::OPENCL; 
            }
            return TargetHint::NEON;
        }
        return target_hint_;
#else
        return TargetHint::NEON;
#endif
    }
    inline void setTargetHint(TargetHint hint){
        target_hint_=hint;
    }
    inline ConvolutionMethodHint & getConvMethod(){ return convolution_method_hint_;}
    inline bool tensor_mem(std::unique_ptr<ACLTensor> &tensor,void *mem){
        tensor->bindmem(mem);
        return true;
    }
    inline bool tensor_mem(void *mem,std::unique_ptr<ACLTensor> &tensor){
        tensor->tensor_copy(mem,false);
        return true;
    }
    bool new_tensor(std::unique_ptr<ACLTensor> &tensor,arm_compute::TensorShape &shape,void *mem=nullptr,bool commit=false);
    bool new_tensor(std::unique_ptr<ACLSubTensor> &tensor,std::unique_ptr<ACLTensor> &parent,arm_compute::TensorShape &shape,arm_compute::Coordinates& coord);
    inline int & group(){return _group;}
    inline void set_operator_property(OperateType type,const char*name){
        name_=name;
        type_=type;
    }
    inline void acl_run(void *input_data, void *output_data){
        if(input_data)tensor_mem(input(),input_data);
        run();
        tensor_mem(output_data,output());
    }


protected:
    inline bool isGPUMode(){
        if (!support_opencl_) return false;
        return getTargetHint()==TargetHint::OPENCL;
    }
    inline OperatorState & opstate(){return operator_state_;}
    inline bool is_operator_init_done(arm_compute::TensorShape shape,TensorType type=tensor_input){
        checkreshape(shape,type);
        return operator_state_==operator_init_done;
    }
    inline void set_operator_init_done(){
        opstate()=operator_init_done;
        set_bypass_state(false);
    }
    inline void set_bypass_state(bool state=false){
        force_bypass_acl_path_=state;
    }
    inline OperatorState checkreshape(arm_compute::TensorShape shape,TensorType type=tensor_input){
        opstate()=reshape(shape,type);
        if (opstate()==operator_reinit) {
            freeres();
        }
        return opstate();
    }
    inline OperatorState reshape(arm_compute::TensorShape &shape,TensorType type){
        arm_compute::TensorShape _shape;
        std::unique_ptr<ACLTensor> &acl_tensor=tensor(type);
        if (!acl_tensor.get()) return operator_not_init;
        _shape = acl_tensor->info().tensor_shape();
        if (_shape.total_size()==shape.total_size() && _shape[0]==shape[0] && _shape[1]==shape[1]) {
            return operator_init_done;
        }
        return operator_reinit;
    }
    inline void freeres(){
        tensor_pool_.clear();
        subtensor_pool_.clear();
        funcs_.clear();
    }
    inline const char* &name(){return name_;}

protected:
    std::vector<std::unique_ptr<TensorPair<ACLTensor>>>tensor_pool_;
    std::vector<std::unique_ptr<TensorPair<ACLSubTensor>>>subtensor_pool_;
    std::vector<std::unique_ptr<arm_compute::IFunction>> funcs_;
    OperatorState operator_state_;
    bool force_bypass_acl_path_;
    TargetHint            target_hint_;             
    ConvolutionMethodHint convolution_method_hint_; 
    static bool support_opencl_;
    static bool init_cl_env;
    int _group;
    const char* name_;
    OperateType type_;
};

int isScheduleEnable();

template <typename OperatorType, typename TensorType>
std::unique_ptr<arm_compute::IFunction> instantiate_function(arm_compute::ITensor *input, arm_compute::ITensor *output){
    auto op = cpp14::make_unique<OperatorType>();
    op->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output)
        );

    return std::move(op);
}

template <typename OperatorType, typename TensorType>
std::unique_ptr<arm_compute::IFunction> instantiate(arm_compute::ITensor *input, arm_compute::ITensor *output)
{
    return instantiate_function<OperatorType, TensorType>(input, output);
}

template <typename GPUOpType,typename GPUTensor,typename CPUOpType,typename CPUTensor>
std::unique_ptr<arm_compute::IFunction> instantiate_op_func(std::unique_ptr<ACLTensor>& input, std::unique_ptr<ACLTensor>& output,TargetHint&  hint){
    std::unique_ptr<arm_compute::IFunction> func;
#ifdef USE_OPENCL
    if(hint == TargetHint::OPENCL)
    {
        func = instantiate<GPUOpType, GPUTensor>(input->tensor(), output->tensor());
    }
    else
#endif
    {
        func = instantiate<CPUOpType, CPUTensor>(input->tensor(), output->tensor());
    }
    return func;
}


template <typename OperatorType, typename TensorType,typename VectorTensor>
std::unique_ptr<arm_compute::IFunction> instantiate_function(VectorTensor inputs, arm_compute::ITensor *output){
    auto op = cpp14::make_unique<OperatorType>();
    op->configure(
        inputs,
        dynamic_cast<TensorType *>(output)
        );

    return std::move(op);
}

template <typename OperatorType, typename TensorType,typename VectorTensor>
std::unique_ptr<arm_compute::IFunction> instantiate(VectorTensor inputs, arm_compute::ITensor *output)
{
    return instantiate_function<OperatorType, TensorType,VectorTensor>(inputs, output);
}

template <typename GPUOpType,typename GPUTensor,typename CPUOpType,typename CPUTensor>
std::unique_ptr<arm_compute::IFunction> instantiate_op_func_lists(ACLOperator*& acl_op, std::unique_ptr<ACLTensor>& output,int num,TargetHint&  hint){
    std::unique_ptr<arm_compute::IFunction> func;
#ifdef USE_OPENCL
    if(hint == TargetHint::OPENCL)
    {
        static std::vector<arm_compute::ICLTensor*> tensors;
        tensors.clear();
        for (int i=0;i<num;++i) {
            tensors.push_back(dynamic_cast<arm_compute::ICLTensor*>(acl_op->cinput(i).get()->tensor()));
        }
        func = instantiate<GPUOpType, GPUTensor, std::vector<arm_compute::ICLTensor *>>(tensors, output->tensor());
    }
    else
#endif
    {
        static std::vector<arm_compute::ITensor*> tensors;
        tensors.clear();
        for (int i=0;i<num;++i) {
            tensors.push_back(dynamic_cast<arm_compute::ITensor*>(acl_op->cinput(i).get()->tensor()));
        }
        func = instantiate<CPUOpType, CPUTensor,std::vector<arm_compute::ITensor*>>(tensors, output->tensor());
    }
    return func;
}

template <typename OperatorType, typename TensorType,typename OperatorInfo>
std::unique_ptr<arm_compute::IFunction> instantiate_function(arm_compute::ITensor *input, arm_compute::ITensor *output, const OperatorInfo &info){
    auto op = cpp14::make_unique<OperatorType>();
    op->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output),
        info);

    return std::move(op);
}

template <typename OperatorType, typename TensorType,typename OperatorInfo>
std::unique_ptr<arm_compute::IFunction> instantiate(arm_compute::ITensor *input, arm_compute::ITensor *output, const OperatorInfo &info)
{
    return instantiate_function<OperatorType, TensorType, OperatorInfo>(input, output, info);
}

template <typename GPUOpType,typename GPUTensor,typename CPUOpType,typename CPUTensor, typename OperatorInfo>
std::unique_ptr<arm_compute::IFunction> instantiate_op_func(std::unique_ptr<ACLTensor>&  input, std::unique_ptr<ACLTensor>&  output, const OperatorInfo &info,TargetHint&  hint){
    std::unique_ptr<arm_compute::IFunction> func;
#ifdef USE_OPENCL
    if(hint == TargetHint::OPENCL)
    {
        func = instantiate<GPUOpType, GPUTensor,OperatorInfo>(input->tensor(), output->tensor(), info);
    }
    else
#endif
    {
        func = instantiate<CPUOpType, CPUTensor,OperatorInfo>(input->tensor(), output->tensor(), info);
    }
    return func;
}


template <typename OperatorType, typename TensorType,typename OperatorInfo>
std::unique_ptr<arm_compute::IFunction> instantiate_function(arm_compute::ITensor *input,arm_compute::ITensor *weights,arm_compute::ITensor *biases, arm_compute::ITensor *output, const OperatorInfo &info){
    auto op = cpp14::make_unique<OperatorType>();
    op->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(weights),
        dynamic_cast<TensorType *>(biases),
        dynamic_cast<TensorType *>(output),
        info);
    return std::move(op);
}

template <typename OperatorType, typename TensorType,typename OperatorInfo>
std::unique_ptr<arm_compute::IFunction> instantiate(arm_compute::ITensor *input,arm_compute::ITensor *weights,arm_compute::ITensor *biases, arm_compute::ITensor *output, const OperatorInfo &info)
{
    return instantiate_function<OperatorType, TensorType, OperatorInfo>(input,weights,biases,output, info);
}

template <typename GPUOpType,typename GPUTensor,typename CPUOpType,typename CPUTensor, typename OperatorInfo,typename ACLTensor>
std::unique_ptr<arm_compute::IFunction> instantiate_op_func(std::unique_ptr<ACLTensor>&  input,std::unique_ptr<ACLTensor>& weights,std::unique_ptr<ACLTensor>&  biases, std::unique_ptr<ACLTensor>&  output, const OperatorInfo &info,TargetHint&  hint){
    std::unique_ptr<arm_compute::IFunction> func;
    arm_compute::ITensor * biases_tensor=NULL;

    if (biases.get()) {
        biases_tensor=biases->tensor();
    }
#ifdef USE_OPENCL
    if (hint == TargetHint::OPENCL)
    {
        func = instantiate<GPUOpType, GPUTensor,OperatorInfo>(input->tensor(), weights->tensor(),biases_tensor,output->tensor(), info);
    }
    else
#endif
    {
        func = instantiate<CPUOpType, CPUTensor,OperatorInfo>(input->tensor(), weights->tensor(),biases_tensor, output->tensor(), info);
    }
    return func;
}



template <typename Dtype,typename OperatorType, typename TensorType>
std::unique_ptr<arm_compute::IFunction> instantiate_function(arm_compute::ITensor *input, arm_compute::ITensor *output,
               arm_compute::ITensor *mean,arm_compute::ITensor *var,arm_compute::ITensor *beta,arm_compute::ITensor *gamma,Dtype & eps){
    auto op = cpp14::make_unique<OperatorType>();
    op->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output),
        dynamic_cast<TensorType *>(mean),
        dynamic_cast<TensorType *>(var),
        dynamic_cast<TensorType *>(beta),
        dynamic_cast<TensorType *>(gamma),
        eps);

    return std::move(op);
}

template <typename Dtype,typename OperatorType, typename TensorType>
std::unique_ptr<arm_compute::IFunction> instantiate(arm_compute::ITensor * input,arm_compute::ITensor * output, 
               arm_compute::ITensor * mean,arm_compute::ITensor * var,arm_compute::ITensor * beta,arm_compute::ITensor * gamma,Dtype eps){
    return instantiate_function<Dtype,OperatorType, TensorType>(input,output, mean,var,beta,gamma,eps);
}

template <typename Dtype,typename GPUOpType,typename GPUTensor,typename CPUOpType,typename CPUTensor>
std::unique_ptr<arm_compute::IFunction> instantiate_op_func(std::unique_ptr<ACLTensor>& input,std::unique_ptr<ACLTensor>& output, 
               std::unique_ptr<ACLTensor>& mean,std::unique_ptr<ACLTensor>& var,std::unique_ptr<ACLTensor>& beta,std::unique_ptr<ACLTensor>& gamma,Dtype eps,TargetHint  hint){
    std::unique_ptr<arm_compute::IFunction> func;
#ifdef USE_OPENCL
    if(hint == TargetHint::OPENCL)
    {
        func = instantiate<Dtype,GPUOpType, GPUTensor>(input->tensor(),output->tensor(), mean->tensor(),var->tensor(),beta->tensor(),gamma->tensor(),eps);
    }
    else
#endif
    {
        func = instantiate<Dtype,CPUOpType, CPUTensor>(input->tensor(),output->tensor(), mean->tensor(),var->tensor(),beta->tensor(),gamma->tensor(),eps);
    }
    return func;
}


template <typename OperatorInfo>
bool instantiate_op_pooling(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input, std::unique_ptr<ACLTensor> & output,TargetHint  hint, const OperatorInfo &info){
    func.push_back(instantiate_op_func<arm_compute::CLPoolingLayer, arm_compute::ICLTensor, arm_compute::NEPoolingLayer, arm_compute::ITensor, arm_compute::PoolingLayerInfo>(input, output, info, hint));
    return true;
}
template <typename OperatorInfo>
bool instantiate_op_activation(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint, const OperatorInfo &info){
    func.push_back(instantiate_op_func<arm_compute::CLActivationLayer,arm_compute::ICLTensor,arm_compute::NEActivationLayer,arm_compute::ITensor, arm_compute::ActivationLayerInfo>(input, output, info, hint));
    return true;
}
template <typename OperatorInfo>
bool instantiate_op_lrn(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint, const OperatorInfo &info){
    func.push_back(instantiate_op_func<arm_compute::CLNormalizationLayer,arm_compute::ICLTensor,arm_compute::NENormalizationLayer,arm_compute::ITensor, arm_compute::NormalizationLayerInfo>(input, output, info, hint));
    return true;
}
template <typename OperatorInfo>
bool instantiate_op_conv(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint,const OperatorInfo &info){
    std::unique_ptr<ACLTensor> & weights=acl_op->weights();
    std::unique_ptr<ACLTensor> & biases=acl_op->biases();
    ConvolutionMethodHint& conv_method=acl_op->getConvMethod();
    bool has_biases=biases.get()?true:false;
    int& groups=acl_op->group();
    arm_compute::TensorShape input_shape=input->info().tensor_shape();
    arm_compute::TensorShape weights_shape=weights->info().tensor_shape();
    arm_compute::TensorShape biases_shape;
    if (has_biases) {
        biases_shape = biases->info().tensor_shape();
    }
    arm_compute::TensorShape output_shape=output->info().tensor_shape();

    if (groups==1) {
        if (conv_method == ConvolutionMethodHint::GEMM) {
            func.push_back(instantiate_op_func<arm_compute::CLConvolutionLayer, arm_compute::ICLTensor, arm_compute::NEConvolutionLayer, arm_compute::ITensor, arm_compute::PadStrideInfo>(acl_op->input(), acl_op->weights(), acl_op->biases(), acl_op->output(), info, hint));
        }else{
            func.push_back(instantiate_op_func<arm_compute::CLDirectConvolutionLayer, arm_compute::ICLTensor, arm_compute::NEDirectConvolutionLayer, arm_compute::ITensor, arm_compute::PadStrideInfo>(acl_op->input(), acl_op->weights(), acl_op->biases(), acl_op->output(), info, hint));
        }
        return true;
    }

    // Calculate sub-tensor splits
    const int input_split   = input_shape.z()  / groups;
    const int output_split  = output_shape.z() / groups;
    const int weights_split = weights_shape[3] / groups;
    const int biases_split  = biases_shape.x() / groups;

    // Calculate sub-tensor shapes
    input_shape.set(2, input_split);
    output_shape.set(2, output_split);
    weights_shape.set(3, weights_split);
    biases_shape.set(0, biases_split);

    for (auto i = 0; i < groups; ++i) {
        // Calculate sub-tensors starting coordinates
        arm_compute::Coordinates input_coord(0, 0, input_split * i);
        arm_compute::Coordinates output_coord(0, 0, output_split * i);
        arm_compute::Coordinates weights_coord(0, 0, 0, weights_split * i);
        arm_compute::Coordinates biases_coord(biases_split * i);

        // Create sub-tensors for input, output, weights and bias
        acl_op->new_tensor(acl_op->sinput(i), acl_op->input(), input_shape, input_coord);
        acl_op->new_tensor(acl_op->soutput(i),acl_op->output(),output_shape, output_coord);
        acl_op->new_tensor(acl_op->sweights(i),acl_op->weights(), weights_shape, weights_coord);
        if (has_biases) {
            acl_op->new_tensor(acl_op->sbiases(i),acl_op->biases(), biases_shape, biases_coord);
        }

        if (conv_method == ConvolutionMethodHint::GEMM) {
            func.push_back(instantiate_op_func<arm_compute::CLConvolutionLayer, arm_compute::ICLTensor, arm_compute::NEConvolutionLayer, arm_compute::ITensor, arm_compute::PadStrideInfo,ACLSubTensor>(acl_op->sinput(i), acl_op->sweights(i), acl_op->sbiases(i), acl_op->soutput(i), info, hint));
        }else{
            func.push_back(instantiate_op_func<arm_compute::CLDirectConvolutionLayer, arm_compute::ICLTensor, arm_compute::NEDirectConvolutionLayer, arm_compute::ITensor, arm_compute::PadStrideInfo,ACLSubTensor>(acl_op->sinput(i), acl_op->sweights(i), acl_op->sbiases(i), acl_op->soutput(i), info, hint));
        }
    }
    return true;
}
template <typename OperatorInfo>
bool instantiate_op_lc(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint, const OperatorInfo &info){
    std::unique_ptr<ACLTensor> & weights=acl_op->weights();
    std::unique_ptr<ACLTensor> & biases=acl_op->biases();
    func.push_back(instantiate_op_func<arm_compute::CLLocallyConnectedLayer,arm_compute::ICLTensor,arm_compute::NELocallyConnectedLayer,arm_compute::ITensor, arm_compute::PadStrideInfo>(input, weights,biases,output,info, hint));
    return true;
}
template <typename OperatorInfo>
bool instantiate_op_fc(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint, const OperatorInfo &info){
    std::unique_ptr<ACLTensor> & weights=acl_op->weights();
    std::unique_ptr<ACLTensor> & biases=acl_op->biases();
    func.push_back(instantiate_op_func<arm_compute::CLFullyConnectedLayer,arm_compute::ICLTensor,arm_compute::NEFullyConnectedLayer,arm_compute::ITensor, bool>(input, weights,biases,output,info, hint));
    return true;
}
template <typename Dtype>
bool instantiate_op_bn(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint, Dtype eps){
    std::unique_ptr<ACLTensor> & mean=acl_op->mean();
    std::unique_ptr<ACLTensor> & var=acl_op->var();
    std::unique_ptr<ACLTensor> & beta=acl_op->beta();
    std::unique_ptr<ACLTensor> & gamma=acl_op->gamma();
    func.push_back(instantiate_op_func<Dtype,arm_compute::CLBatchNormalizationLayer,arm_compute::ICLTensor,arm_compute::NEBatchNormalizationLayer,arm_compute::ITensor>(input, output, mean,var,beta,gamma,eps, hint));
    return true;
}
inline bool instantiate_op_softmax(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint,void *data){
    func.push_back(instantiate_op_func<arm_compute::CLSoftmaxLayer,arm_compute::ICLTensor,arm_compute::NESoftmaxLayer,arm_compute::ITensor>(input, output, hint));
    return true;
}
inline bool instantiate_op_concat(ACLOperator* acl_op,std::vector<std::unique_ptr<arm_compute::IFunction>> & func,std::unique_ptr<ACLTensor> & input,std::unique_ptr<ACLTensor> & output,TargetHint  hint,int num){
    func.push_back(instantiate_op_func_lists<arm_compute::CLDepthConcatenateLayer,arm_compute::ICLTensor,arm_compute::NEDepthConcatenateLayer,arm_compute::ITensor>(acl_op, output, num,hint));
    return true;
}
template <typename Dtype>
Dtype* GetDataPtr(ACLOperator* op,Blob<Dtype>* const &blob,bool isconst=false){
    if (!isconst) {
        if (op->getTargetHint() == TargetHint::NEON) {
            return blob->mutable_cpu_data();
        }
        return blob->mutable_gpu_data();
    }
    if (op->getTargetHint()==TargetHint::NEON) {
        return (Dtype*)blob->cpu_data();
    }
    return (Dtype*)blob->gpu_data();
}

template <typename Dtype>
Dtype* InputdataPtr(ACLOperator* op,const vector<Blob<Dtype>*>& bottom,int index=-1){
    if (index==-1) index=0;
    return GetDataPtr(op, bottom[index], true);
}
template <typename Dtype>
Dtype* OutputdataPtr(ACLOperator* op,const vector<Blob<Dtype>*>& top){
    return GetDataPtr(op,top[0]);
}

template <typename Dtype>
void acl_run(ACLOperator* op,const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top,bool multi_input_run=true){
    if (multi_input_run) {
        for (int i = 0; i < bottom.size(); ++i) {
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* top_data = top[i]->mutable_cpu_data();
            op->acl_run((void*)bottom_data,(void*)top_data);
        }
        return ;
    }
    for (int i = 0; i < bottom.size(); ++i) {
        op->tensor_mem(op->cinput(i),InputdataPtr(op,bottom,i));
    }
    op->acl_run(NULL,OutputdataPtr(op,top));
}
}

#define INIT_GLOBAL_FUNCS_TYPE(Dtype) \
template <> \
Dtype* InputdataPtr(ACLOperator* op,const vector<Blob<Dtype>*>& bottom,int index); \
template <> \
Dtype* OutputdataPtr(ACLOperator* op,const vector<Blob<Dtype>*>& top); \
template <> \
Dtype* GetDataPtr(ACLOperator* op,Blob<Dtype>* const & blob,bool isconst); \

#define INIT_GLOBAL_FUNCS() \
INIT_GLOBAL_FUNCS_TYPE(double); \
INIT_GLOBAL_FUNCS_TYPE(float); \


#ifdef USE_PROFILING 
#define acl_configure(opname,acl_op,args...)\
{\
            set_operator_property(operate_type_##opname,#opname); \
            logtime_util log_time(ACL_CONFIG_INFO); \
            instantiate_op_##opname(acl_op,acl_op->funcs(),acl_op->input(),acl_op->output(),acl_op->getTargetHint(),args);\
}
#else
#define acl_configure(opname,acl_op,args...)\
{\
            set_operator_property(operate_type_##opname,#opname); \
            instantiate_op_##opname(acl_op,acl_op->funcs(),acl_op->input(),acl_op->output(),acl_op->getTargetHint(),args);\
}
#endif 

#endif

#endif
