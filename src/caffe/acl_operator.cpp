#ifdef USE_ACL
#include "caffe/acl_operator.hpp"
#include "caffe/common.hpp"

unsigned int bypass_acl_class_layer =    (0 | \
                                          FLAGS_ENABLE_ACL_CONCAT | \
                                          /*0xffffffff |*/ \
                                          /*FLAGS_ENABLE_ACL_FC |*/ \
                                          /*FLAGS_ENABLE_ACL_LRN |*/ \
                                          0 );

unsigned int openailab_intfp   = 0;
int enable_schedule=0;

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
bool AclEnableSchedule(int enable){
    enable_schedule=enable;
    if (enable) {
        Caffe::set_mode(Caffe::GPU);
    }
    return true;
}
int isScheduleEnable()
{
    return enable_schedule;
}
bool ACLOperator::init_cl_env=true;
bool ACLOperator::support_opencl_=false;
bool opencl_is_available()
{
    return arm_compute::opencl_is_available();
}
ACLOperator::ACLOperator(const LayerParameter& param)
    :operator_state_(operator_not_init),force_bypass_acl_path_(false),
    target_hint_(TargetHint::DONT_CARE),
    convolution_method_hint_(ConvolutionMethodHint::GEMM),
    _group(1),name_(""){
  const char* pBypassACL;
  if(init_cl_env){
#ifdef USE_OPENCL
     try {
        if (opencl_is_available()) {
          arm_compute::CLScheduler::get().default_init();
          support_opencl_=true;
        }
     }catch(std::exception& e){
          support_opencl_=false;
     }
#endif
     init_cl_env=false;
  }
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

  const string& layer_type = param.type();
  if (layer_type=="Convolution") {
      ConvolutionParameter conv_param = param.convolution_param();
        const char* pDirectConv;
        unsigned int use_direct_conv=0;
        pDirectConv = getenv ("DIRECTCONV");
        if (pDirectConv){
          unsigned int bdirectconv;
          sscanf(pDirectConv,"%i", &bdirectconv);
          if(bdirectconv != use_direct_conv){
              use_direct_conv = bdirectconv;
              printf("DIRECTCONV<%s>\n", pDirectConv);
              printf("DIRECTCONV: %x\n", use_direct_conv);
          }
        }
        int pad_data[3];
        if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
          pad_data[0] = conv_param.pad_h();
          pad_data[1] = conv_param.pad_w();
        } else {
          const int kDefaultPad = 0;
          const int num_pad_dims = conv_param.pad_size();
          for (int i = 0; i < 2; ++i) {
            pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                conv_param.pad((num_pad_dims == 1) ? 0 : i);
          }
        }
        if (use_direct_conv && ( (conv_param.kernel_size(0)==1 &&pad_data[0]==0 && pad_data[1]==0) || (conv_param.kernel_size(0)==3 && pad_data[0]<=1 && pad_data[1] <=1 ) )) {
            convolution_method_hint_=ConvolutionMethodHint::DIRECT; //NEDirectConvolutionLayer only for 1x1 and 3x3
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
  const char* pEnableSchedule;
  pEnableSchedule = getenv ("ENABLESCHEDULE");
  if (pEnableSchedule){
    unsigned int bshedule;
    sscanf(pEnableSchedule,"%i", &bshedule);
    if(bshedule != enable_schedule){
        enable_schedule = bshedule;
        printf("ENABLESCHEDULE<%s>\n", pEnableSchedule);
        printf("ENABLESCHEDULE: %x\n", enable_schedule);
    }
    if (enable_schedule) {
        AclEnableSchedule(1);
    }
  }
}
ACLOperator::~ACLOperator() {
}

bool ACLOperator::new_tensor(std::unique_ptr<ACLTensor> &tensor,arm_compute::TensorShape &shape,void *mem,bool commit)
{
    auto acl_tensor=new ACLTensor(arm_compute::TensorInfo(shape, arm_compute::Format::F32));
    acl_tensor->set_target(getTargetHint());
    acl_tensor->bindmem(mem);
    if (commit) acl_tensor->commit();
    tensor=(std::unique_ptr<ACLTensor>) std::move(acl_tensor);
    return true;
}
bool ACLOperator::new_tensor(std::unique_ptr<ACLSubTensor> &tensor,std::unique_ptr<ACLTensor> &parent,arm_compute::TensorShape &shape,arm_compute::Coordinates& coord)
{
    auto acl_tensor=new ACLSubTensor(parent,shape, coord);
    acl_tensor->set_target(getTargetHint());
    tensor=(std::unique_ptr<ACLSubTensor>) std::move(acl_tensor);
    return true;
}

void ACLTensor::commit(TensorType type)
{
    settensortype(type);
    if (mem_) {
        if (!allocate_){ 
#ifdef USE_PROFILING
            logtime_util log_time(ACL_ALLOCATE_INFO);
#endif //USE_PROFILING
            allocate(); 
            allocate_=true;
        }
        if (type_!= tensor_output) {
           tensor_copy(mem_);
        }
        mem_=nullptr;
    }
}

int BaseACLTensor::tensor_copy(arm_compute::ITensor* tensor,void * mem,bool toTensor)
{
#ifdef USE_PROFILING
    logtime_util log_time(ACL_COPY_INFO);
#endif //USE_PROFILING
    arm_compute::Window window;
    window.use_tensor_dimensions(tensor->info()->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY); // Iterate through the rows (not each element)
    int width = tensor->info()->tensor_shape()[0]; 
    int height = tensor->info()->tensor_shape()[1];
    int deepth = tensor->info()->tensor_shape()[2];
    map();
    // Create an iterator:
    arm_compute::Iterator it(tensor, window);
    // Except it works for an arbitrary number of dimensions
    if (toTensor) { //mem->tensor
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
        {
                memcpy(it.ptr(), ((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width + id.x()) * tensor->info()->element_size()), width * tensor->info()->element_size());
        },
        it);
    }else{ //tensor-->mem
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
        {
                memcpy(((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width) * tensor->info()->element_size()), it.ptr(), width * tensor->info()->element_size());
        },
        it);
    }
    unmap();

    return 0;
}

INIT_GLOBAL_FUNCS();

}


#endif
