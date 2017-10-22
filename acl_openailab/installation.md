# 1. User Quick Guide
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This Installation will help you get started to setup CaffeOnACL on RK3399 quickly.

# 2. Preparation
## 2.1 General dependencies installation
	sudo apt-get -y update
	sodo apt-get -y upgrade
	sudo apt-get install -y build-essential pkg-config automake autoconf protobuf-compiler cmake cmake-gui
	sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev
	sudo apt-get install -y libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libopenblas-dev
	sudo apt-get install -y libopencv-dev python-dev
	sudo apt-get install -y python-numpy python-scipy python-yaml python-six python-pip
	sudo apt-get install -y scons git
	sudo apt-get install -y --no-install-recommends libboost-all-dev
	pip install --upgrade pip

## 2.2 Download source code

	cd ~

#### Download "ACL" (arm_compute :[v17.10](https://github.com/ARM-software/ComputeLibrary/tree/bf8b01dfbfdca124673ade33c5eac8f3748d7abd)):
	git clone https://github.com/ARM-software/ComputeLibrary.git
	git checkout bf8b01d
#### Download "CaffeOnACL" :
	git clone https://github.com/OAID/CaffeOnACL.git
#### Download "Googletest" :
	git clone https://github.com/google/googletest.git

# 3. Build CaffeOnACL
## 3.1 Build ACL :
	cd ~/ComputeLibrary
    aarch64-linux-gnu-gcc opencl-1.2-stubs/opencl_stubs.c -Iinclude -shared -o build/libOpenCL.so
	scons Werror=1 -j8 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a

## 3.2 Build Caffe :
	export ACL_ROOT=~/ComputeLibrary
	cd ~/CaffeOnACL
	cp Makefile.config.acl Makefile.config
	make all distribute

## 3.3 Build Unit tests
##### Build the gtest libraries
	cd ~/googletest
	cmake CMakeLists.txt
	make
	sudo make install

##### Build Caffe Unit tests
	export CAFFE_ROOT=~/CaffeOnACL
	cd ~/CaffeOnACL/unit_tests
	make clean
	make

## 3.4 To Configure The Libraries

	sudo cp ~/ComputeLibrary/build/libarm_compute.so /usr/lib 
	sudo cp ~/ComputeLibrary/build/libarm_compute_core.so /usr/lib 
	sudo cp ~/CaffeOnACL/distribute/lib/libcaffe.so  /usr/lib

# 4. Run tests

#### 4.1 Run Caffenet
	cd  ~/CaffeOnACL/data/ilsvrc12
	sudo chmod +x get_ilsvrc_aux.sh
	./get_ilsvrc_aux.sh
	cd ../..
	./scripts/download_model_binary.py ./models/bvlc_reference_caffenet
	./distribute/bin/classification.bin models/bvlc_reference_caffenet/deploy.prototxt models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg
	output message --
	  ---------- Prediction for examples/images/cat.jpg ----------
	  0.3094 - "n02124075 Egyptian cat"
	  0.1761 - "n02123159 tiger cat"
	  0.1221 - "n02123045 tabby, tabby cat"
	  0.1132 - "n02119022 red fox, Vulpes vulpes"
	  0.0421 - "n02085620 Chihuahua"

#### 4.2 Run Unit test
	  cd ~/CaffeOnACL/unit_tests
	  ./test_caffe_main
	  output message:
	    [==========] 29 tests from 6 test cases ran. (1236 ms total) [ PASSED ] 29 tests.
