#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#include <glog/logging.h>

extern "C" {
#include "testbed.h"
}


#define TYPED_TEST(a,b) template <typename TypeParam> void a <TypeParam>:: b (void)
#define EXPECT_NEAR(a,b,c) {}
#define EXPECT_EQ(a,b) {}

namespace caffe {

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};


template <typename TypeParam>
class ConvolutionLayerTest {
  typedef typename TypeParam::Dtype Dtype;

public:

  void TestSimpleConvolution(void);

  void TestDilatedConvolution(void);

  void Test0DConvolution(void);

  void TestSimple3DConvolution(void);

  void TestDilated3DConvolution(void);

  void Test1x1Convolution(void);

  void TestSimpleConvolutionGroup(void);
  
  void TestNDAgainst2D(void);

  void RunConvolution(void);

  ConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Layer<Dtype> > layer;
};

TYPED_TEST(ConvolutionLayerTest, RunConvolution) {

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  layer=shared_ptr<Layer<Dtype> > (new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(ConvolutionLayerTest, TestDilatedConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> bottom_shape;
  bottom_shape.push_back(2);
  bottom_shape.push_back(3);
  bottom_shape.push_back(8);
  bottom_shape.push_back(7);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_dilation(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  layer=shared_ptr<Layer<Dtype> > (new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ConvolutionLayerTest, Test0DConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  const int kNumOutput = 3;
  convolution_param->set_num_output(kNumOutput);
  convolution_param->set_axis(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  layer=shared_ptr<Layer<Dtype> > (
      new ConvolutionLayer<Dtype>(layer_param));
  vector<int> top_shape = this->blob_bottom_->shape();
  top_shape[3] = kNumOutput;
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(top_shape, this->blob_top_->shape());
}

TYPED_TEST(ConvolutionLayerTest, TestSimple3DConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 5;
  bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
  bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  layer=shared_ptr<Layer<Dtype> > (
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ConvolutionLayerTest, TestDilated3DConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 6;
  bottom_shape[3] = 7;
  bottom_shape[4] = 8;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_dilation(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  layer=shared_ptr<Layer<Dtype> > (
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ConvolutionLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  layer=shared_ptr<Layer<Dtype> > (
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  layer=shared_ptr<Layer<Dtype> > (
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe

using namespace caffe;
ConvolutionLayerTest<CPUDevice<float> > * g_convptr;

void single_forward(void * dummy )
{
  g_convptr->RunConvolution();
}

void forward_convolution(void)
{
   run_test(16,0,single_forward,NULL);
}

#define RUN_FUNC(test_case) test_ ## test_case ()

#define DEF_TEST_FUNC(test_case) \
void test_## test_case (void)\
{\
   std::cout<<__FUNCTION__<<"  start ..."<<std::endl;\
   g_convptr=new ConvolutionLayerTest<CPUDevice<float> >;\
   g_convptr->SetUp();\
   g_convptr->Test ## test_case ();\
   forward_convolution();\
   delete  g_convptr;\
   std::cout<<__FUNCTION__<<"  DONE"<<std::endl;\
}

DEF_TEST_FUNC(SimpleConvolution)
DEF_TEST_FUNC(DilatedConvolution)
DEF_TEST_FUNC(0DConvolution)
DEF_TEST_FUNC(Simple3DConvolution)
DEF_TEST_FUNC(Dilated3DConvolution)
DEF_TEST_FUNC(1x1Convolution)
DEF_TEST_FUNC(SimpleConvolutionGroup)


int main(int argc, char * argv[])
{
    caffe::GlobalInit(&argc, &argv);

    init_testbed();

    RUN_FUNC(SimpleConvolution);
    RUN_FUNC(DilatedConvolution);
    RUN_FUNC(0DConvolution);
    RUN_FUNC(Simple3DConvolution);
    RUN_FUNC(Dilated3DConvolution);
    RUN_FUNC(1x1Convolution);
    RUN_FUNC(SimpleConvolutionGroup);

    release_testbed();
    return 0;
}
