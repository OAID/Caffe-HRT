#ifndef CAFFE_ACL_BASE_CONV_LAYER_HPP_
#define CAFFE_ACL_BASE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

#ifdef USE_ACL
#include "caffe/acl_layer.hpp"
#endif

namespace caffe {

#ifdef USE_ACL
/*
 * @brief ACL implementation of ConvolutionLayer.
 *        Fallback to ConvolutionLayer for some corner cases.
 *
*/
template <typename Dtype,typename GPUConvLayer,typename CPUConvLayer>
class ACLConvolutionLayer : public ACLBaseLayer<GPUConvLayer,CPUConvLayer>,public ConvolutionLayer<Dtype> {
 public:
  explicit ACLConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~ACLConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		  NOT_IMPLEMENTED;
      }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		  NOT_IMPLEMENTED;
      }
  virtual void SetupACLLayer(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

};
#endif

}  // namespace caffe

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CONV_CLASS(classname,GPUConvLayer,CPUConvLayer) \
  template class classname<float,GPUConvLayer,CPUConvLayer>; \
  template class classname<double,GPUConvLayer,CPUConvLayer>

#endif  // CAFFE_ACL_BASE_CONV_LAYER_HPP_
