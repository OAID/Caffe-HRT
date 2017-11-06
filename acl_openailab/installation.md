# 1. User Quick Guide
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This Installation will help you get started to setup CaffeOnACL on RK3399 quickly.

# 2. Preparation
## 2.1 General dependencies installation
	sudo apt-get install git cmake scons protobuf-compiler libgflags-dev libgoogle-glog-dev 
	sudo apt-get install libblas-dev libhdf5-serial-dev liblmdb-dev libleveldb-dev 
	sudo apt-get install liblapack-dev libsnappy-dev python-numpy 
	sudo apt-get install libprotobuf-dev libopenblas-dev libgtk2.0-dev
	sudo apt-get install python-yaml python-numpy python-scipy python-six
	
	sudo apt-get install --no-install-recommends libboost-all-dev

## 2.2 Download source code

	cd ~
	
#### Download "OpenCV" 
	wget --no-check-certificate https://github.com/opencv/opencv/archive/3.3.0.tar.gz
	tar -xvf 3.3.0.tar.gz
#### Download "gen-pkg-config-pc" 
	wget https://github.com/OAID/AID-tools/tree/master/script/gen-pkg-config-pc.sh
#### Download "ACL" 
	git clone https://github.com/ARM-software/ComputeLibrary.git
#### Download "CaffeOnACL" :
	git clone https://github.com/OAID/CaffeOnACL.git
#### Download "Googletest" :
	git clone https://github.com/google/googletest.git

# 3. Build CaffeOnACL

## 3.1 Build OpenCV :
	cd ~/opencv-3.3.0
	mkdir build
	cd build
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/AID/opencv3.3.0 ..
	sudo make install
	sudo ~/gen-pkg-config-pc.sh /usr/local/AID
	
## 3.2 Build ACL :
	cd ~/ComputeLibrary
    aarch64-linux-gnu-gcc opencl-1.2-stubs/opencl_stubs.c -Iinclude -shared -o build/libOpenCL.so
	scons Werror=1 -j4 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a
	wget https://github.com/OAID/AID-tools/tree/master/script/Computelibrary/Makefile
	sudo make install

## 3.3 Build Caffe :
	cd ~/CaffeOnACL
	make all 
	make distribute
	sudo make install

## 3.4 Build Unit tests
##### Build the gtest libraries
	cd ~/googletest
	cmake -D CMAKE_INSTALL_PREFIX=/usr/local/AID/googletest CMakeLists.txt
	make
	sudo make install

##### Build Caffe Unit tests
	cd ~/CaffeOnACL/unit_tests
	make clean
	make

## 3.5 To Configure The Libraries
	sudo ~/gen-pkg-config-pc.sh /usr/local/AID

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
