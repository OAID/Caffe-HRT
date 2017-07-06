![OPEN AI LAB](https://oaid.github.io/pics/openailab.png)

# 1. Release Notes
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Please refer to [CaffeOnACL Release NOTE](https://github.com/OAID/caffeOnACL/blob/master/acl_openailab/Reversion.md) for details

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
Recommend creating a new directory in your work directory to execute the following steps. For example, you can create a direcotry named "oaid" in your home directory by the following commands.<br>

	cd ~
	mkdir oaid
	cd oaid

#### Download "ACL" (arm_compute : v17.05):
	git clone https://github.com/ARM-software/ComputeLibrary.git
#### Download "CaffeOnACL" :
	git clone https://github.com/OAID/caffeOnACL.git
#### Download "Googletest" :
	git clone https://github.com/google/googletest.git

# 3. Build CaffeOnACL
## 3.1 Build ACL :
	cd ~/oaid/ComputeLibrary
	scons Werror=1 -j8 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a

## 3.2 Build Caffe :
	export ACL_ROOT=~/oaid/ComputeLibrary
	cd ~/oaid/caffeOnACL
	cp acl_openailab/Makefile.config.acl Makefile.config
	make all distribute

## 3.3 Build Unit tests
##### Build the gtest libraries
	cd ~/oaid/googletest
	cmake CMakeLists.txt
	make
	sudo make install

##### Build Caffe Unit tests
	export CAFFE_ROOT=~/oaid/caffeOnACL
	cd ~/oaid/caffeOnACL/unit_tests
	make clean
	make

## 3.3 Run tests
If the output message of the following two tests is same as the examples, it means the porting is success.

	export LD_LIBRARY_PATH=~/oaid/caffeOnACL/distribute/lib:~/oaid/ComputeLibrary/build

#### Reference Caffenet
	cd  ~/oaid/caffeOnACL/data/ilsvrc12
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

#### Unit test
	  cd ~/oaid/caffeOnACL/unit_tests
	  ./test_caffe_main
	  output message:
	    [==========] 29 tests from 6 test cases ran. (1236 ms total) [ PASSED ] 29 tests.
