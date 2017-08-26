#ifndef CAFFE_ACL_CONV_LAYER_HPP_
#define CAFFE_ACL_CONV_LAYER_HPP_

#ifdef USE_ACL
#include "caffe/layers/acl_base_conv_layer.hpp"
#endif

namespace caffe {

extern bool use_direct_conv_;
#ifdef USE_ACL
template <typename Dtype>
inline shared_ptr<Layer<Dtype> > GetACLConvolutionLayer(
    const LayerParameter& param) {
    ConvolutionParameter conv_param = param.convolution_param();
    const char* pDirectConv;
    pDirectConv = getenv ("DIRECTCONV");
    if (pDirectConv){
      unsigned int bdirectconv;
      sscanf(pDirectConv,"%i", &bdirectconv);
      if(bdirectconv != use_direct_conv_){
          use_direct_conv_ = bdirectconv;
          printf("DIRECTCONV<%s>\n", pDirectConv);
          printf("DIRECTCONV: %x\n", use_direct_conv_);
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
    if (use_direct_conv_ && ( (conv_param.kernel_size(0)==1 &&pad_data[0]==0 && pad_data[1]==0) || (conv_param.kernel_size(0)==3 && pad_data[0]<=1 && pad_data[1] <=1 ) )) {
        return shared_ptr<Layer<Dtype> >(new ACLConvolutionLayer<Dtype, CLConvolutionLayer, NEDirectConvolutionLayer>(param)); //NEDirectConvolutionLayer only for 1x1 and 3x3
    }
    return shared_ptr<Layer<Dtype> >(new ACLConvolutionLayer<Dtype, CLConvolutionLayer, NEConvolutionLayer>(param)); 
}
#endif

}  // namespace caffe

#endif  // CAFFE_ACL_CONV_LAYER_HPP_
