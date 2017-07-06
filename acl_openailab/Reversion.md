# Release Note
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

The release version is 0.2.0. You can download the source code from [OAID/caffeOnACL](https://github.com/OAID/caffeOnACL)

## Verified Platform :

The release is verified on 64bits ARMv8 processor<br>
* Hardware platform : Rockchip RK3399 (firefly RK3399 board)<br>
* Software platform : Ubuntu 16.04<br>

## 10 Layers accelerated by ACL layers :
* ConvolutionLayer
* PoolingLayer
* LRNLayer
* ReLULayer
* SigmoidLayer
* SoftmaxLayer
* TanHLayer
* AbsValLayer
* BNLLLayer
* InnerProductLayer

## ACL compatibility issues :
There are some compatibility issues between ACL and caffe Layers, we bypass it to Caffe's original layer class as the workaround solution for the below issues
* Normalization in-channel issue
* Tanh issue
* Even Kernel size
* Softmax supporting multi-dimension issue
* Group issue
* Performance need be fine turned in the future

# Changelist
The caffe based version is `793bd96351749cb8df16f1581baf3e7d8036ac37`.
## New Files :
	Makefile.config.acl
	cmake/Modules/FindACL.cmake
	examples/cpp_classification/classification_profiling.cpp
	examples/cpp_classification/classification_profiling_gpu.cpp
	include/caffe/acl_layer.hpp
	include/caffe/layers/acl_absval_layer.hpp
	include/caffe/layers/acl_base_activation_layer.hpp
	include/caffe/layers/acl_bnll_layer.hpp
	include/caffe/layers/acl_conv_layer.hpp
	include/caffe/layers/acl_inner_product_layer.hpp
	include/caffe/layers/acl_lrn_layer.hpp
	include/caffe/layers/acl_pooling_layer.hpp
	include/caffe/layers/acl_relu_layer.hpp
	include/caffe/layers/acl_sigmoid_layer.hpp
	include/caffe/layers/acl_softmax_layer.hpp
	include/caffe/layers/acl_tanh_layer.hpp
	models/SqueezeNet/README.md
	models/SqueezeNet/SqueezeNet_v1.1/squeezenet.1.1.deploy.prototxt
	src/caffe/acl_layer.cpp
	src/caffe/layers/acl_absval_layer.cpp
	src/caffe/layers/acl_base_activation_layer.cpp
	src/caffe/layers/acl_bnll_layer.cpp
	src/caffe/layers/acl_conv_layer.cpp
	src/caffe/layers/acl_inner_product_layer.cpp
	src/caffe/layers/acl_lrn_layer.cpp
	src/caffe/layers/acl_pooling_layer.cpp
	src/caffe/layers/acl_relu_layer.cpp
	src/caffe/layers/acl_sigmoid_layer.cpp
	src/caffe/layers/acl_softmax_layer.cpp
	src/caffe/layers/acl_tanh_layer.cpp
	unit_tests/Makefile
	unit_tests/pmu.c
	unit_tests/pmu.h
	unit_tests/prof_convolution_layer.cpp
	unit_tests/sgemm.cpp
	unit_tests/test.cpp
	unit_tests/test_caffe_main.cpp
	unit_tests/test_common.cpp
	unit_tests/test_convolution_layer.cpp
	unit_tests/test_fail.cpp
	unit_tests/test_inner_product_layer.cpp
	unit_tests/test_lrn_layer.cpp
	unit_tests/test_neuron_layer.cpp
	unit_tests/test_pooling_layer.cpp
	unit_tests/test_softmax_layer.cpp
	unit_tests/testbed.c
	unit_tests/testbed.h

## Change Files :
	Makefile
	cmake/Dependencies.cmake
	include/caffe/caffe.hpp
	include/caffe/common.hpp
	include/caffe/layer.hpp
	include/caffe/util/device_alternate.hpp
	include/caffe/util/hdf5.hpp
	src/caffe/common.cpp
	src/caffe/layer.cpp
	src/caffe/layer_factory.cpp
	src/caffe/layers/absval_layer.cpp
	src/caffe/layers/bnll_layer.cpp
	src/caffe/layers/hdf5_data_layer.cpp
	src/caffe/layers/hdf5_data_layer.cu
	src/caffe/layers/hdf5_output_layer.cpp
	src/caffe/layers/hdf5_output_layer.cu
	src/caffe/layers/inner_product_layer.cpp
	src/caffe/net.cpp
	src/caffe/solvers/sgd_solver.cpp
	src/caffe/syncedmem.cpp
	src/caffe/test/test_hdf5_output_layer.cpp
	src/caffe/test/test_hdf5data_layer.cpp
	src/caffe/util/hdf5.cpp
	src/caffe/util/math_functions.cpp

# Issue report
Encounter any issue, please report on [issue report](https://github.com/OAID/caffeOnACL/issues). Issue report should contain the following information :
* The exact description of the steps that are needed to reproduce the issue 
* The exact description of what happens and what you think is wrong 
