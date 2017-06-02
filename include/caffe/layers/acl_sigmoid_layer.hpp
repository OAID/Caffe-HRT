#ifndef CAFFE_ACL_SIGMOID_LAYER_HPP_
#define CAFFE_ACL_SIGMOID_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

#ifdef USE_ACL
#include "caffe/acl_layer.hpp"
#include "caffe/layers/acl_base_activation_layer.hpp"
#endif

namespace caffe {

#ifdef USE_ACL
/**
 * @brief ACL acceleration of SigmoidLayer.
 */
template <typename Dtype>
class ACLSigmoidLayer : public ACLBaseActivationLayer<Dtype>,public SigmoidLayer<Dtype> {
 public:
  explicit ACLSigmoidLayer(const LayerParameter& param)
      : ACLBaseActivationLayer<Dtype>(param),SigmoidLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~ACLSigmoidLayer();

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
      const vector<Blob<Dtype>*>& top, ActivationLayerInfo::ActivationFunction type);
};
#endif

}  // namespace caffe

#endif  // CAFFE_ACL_SIGMOID_LAYER_HPP_
