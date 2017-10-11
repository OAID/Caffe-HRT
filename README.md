

# CaffeOnACL
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

CaffeOnACL is a project that is maintained by **OPEN** AI LAB, it uses Arm Compute Library (NEON+GPU) to speed up [Caffe](http://caffe.berkeleyvision.org/) and provide utilities to debug, profile and tune application performance. 

The release version is 0.4.0, is based on [Rockchip RK3399](http://www.rock-chips.com/plus/3399.html) Platform, target OS is Ubuntu 16.04. Can download the source code from [OAID/CaffeOnACL](https://github.com/OAID/CaffeOnACL)

* The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. See also [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary).
* Caffe is a fast open framework for deep learning. See also [Caffe](https://github.com/BVLC/caffe).

### Documents
* [Installation instructions](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/installation.md)
* [User Manuals PDF](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/user_manual.pdf)
* [Performance Report PDF](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/performance_report.pdf)

### Arm Compute Library Compatibility Issues :
There are some compatibility issues between ACL and Caffe Layers, we bypass it to Caffe's original layer class as the workaround solution for the below issues

* Normalization in-channel issue
* Tanh issue
* Softmax supporting multi-dimension issue
* Group issue

Performance need be fine turned in the future

# Release History
The Caffe based version is [793bd96351749cb8df16f1581baf3e7d8036ac37](https://github.com/BVLC/caffe/tree/793bd96351749cb8df16f1581baf3e7d8036ac37).


### Version 0.4.0 - Oct 11, 2017
Support Arm Compute Library version 17.09

### Version 0.3.0 - Aug 26, 2017
Support Arm Compute Library version 17.06 with 4 new layers added

* Batch Normalization Layer
* Direct convolution Layer
* Locally Connect Layer
* Concatenate layer


### Version 0.2.0 - Jul 2, 2017

Fix the issues:

* Compatible with Arm Compute Library version 17.06
* When OpenCL initialization fails, even if Caffe uses CPU-mode,it doesn't work properly.

### Version 0.1.0 - Jun 2, 2017 
   
  Initial version supports 10 Layers accelerated by Arm Compute Library version 17.05 : 

* Convolution Layer
* Pooling Layer
* LRN Layer
* ReLU Layer
* Sigmoid Layer
* Softmax Layer
* TanH Layer
* AbsVal Layer
* BNLL Layer
* InnerProduct Layer

# Issue Report
Encounter any issue, please report on [issue report](https://github.com/OAID/CaffeOnACL/issues). Issue report should contain the following information :

*  The exact description of the steps that are needed to reproduce the issue 
* The exact description of what happens and what you think is wrong 
