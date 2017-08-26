#ifndef CAFFE_LOCALCONNECT_LAYERS_HPP_
#define CAFFE_LOCALCONNECT_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
class LocalConnectLayer : public Layer<Dtype> {
 public:
  explicit LocalConnectLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  int kernel_size_;
  int stride_;
  int dilation_;
  int num_;
  int channels_;
  int pad_;
  int height_, width_;
  int height_out_, width_out_;
  int num_output_;
  bool bias_term_;

  int M_;
  int K_;
  int N_;

  Blob<Dtype> col_buffer_;
};
}  // namespace caffe

#endif  // CAFFE_LOCALCONNECT_LAYERS_HPP_
