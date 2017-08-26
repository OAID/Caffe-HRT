#ifndef CAFFE_ACL_LOCALCONNECT_LAYER_HPP_
#define CAFFE_ACL_LOCALCONNECT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/local_connect_layer.hpp"

#ifdef USE_ACL
#include "caffe/acl_layer.hpp"
#endif

namespace caffe {

#ifdef USE_ACL
/*
 * @brief ACL implementation of LocalConnectLayer.
 *        Fallback to LocalConnectLayer for some corner cases.
 *
*/
template <typename Dtype>
class ACLLocalConnectLayer : public ACLBaseLayer<CLLocallyConnectedLayer,NELocallyConnectedLayer>,public LocalConnectLayer<Dtype> {
 public:
  explicit ACLLocalConnectLayer(const LayerParameter& param)
      : LocalConnectLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~ACLLocalConnectLayer();

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

#endif  // CAFFE_ACL_LOCALCONNECT_LAYER_HPP_
