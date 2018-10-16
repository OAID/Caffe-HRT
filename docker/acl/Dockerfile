FROM ubuntu:16.04

WORKDIR /

RUN mkdir /build \
    && cd /build \
    && apt-get update \
    && apt-get install -y wget cmake git scons protobuf-compiler libgflags-dev libgoogle-glog-dev \
    && apt-get install -y libblas-dev libhdf5-serial-dev liblmdb-dev libleveldb-dev apt-utils \
    && apt-get install -y liblapack-dev libsnappy-dev python-dev python-pip \
    && apt-get install -y libprotobuf-dev libopenblas-dev libgtk2.0-dev \
    && apt-get install -y python-yaml python-numpy python-scipy python-six \
    && pip install pyyaml numpy scipy six "matplotlib>=2.0.0,<3.0.0" \
    && apt-get install -y --no-install-recommends libboost-all-dev \
    && git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/cachengo/Caffe-HRT.git \
    && wget ftp://ftp.openailab.net/tools/script/gen-pkg-config-pc.sh \
    && chmod +x /build/gen-pkg-config-pc.sh \
    && git clone https://github.com/ARM-software/ComputeLibrary.git \
    && git clone https://github.com/google/googletest.git \
    && cd /build/opencv \
    && git checkout 3.3.0 \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/AID/opencv3.3.0 .. \
    && make install \
    && /build/gen-pkg-config-pc.sh /usr/local/AID \
    && cd /build/ComputeLibrary \
    && git checkout 48bc34e \
    && mkdir build \
    && aarch64-linux-gnu-gcc opencl-1.2-stubs/opencl_stubs.c -Iinclude -shared -o build/libOpenCL.so \
    && scons Werror=1 -j4 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a \
    && wget ftp://ftp.openailab.net/tools/script/Computelibrary/Makefile \
    && make install \
    && /build/gen-pkg-config-pc.sh /usr/local/AID \
    && cd /build/Caffe-HRT \
    && apt install -y ocl-icd-opencl-dev \
    && make all \
    && make distribute \
    && make install \
    && /build/gen-pkg-config-pc.sh /usr/local/AID \
    && cd /build/googletest \
    && cmake -D CMAKE_INSTALL_PREFIX=/usr/local/AID/googletest CMakeLists.txt \
    && make \
    && make install \
    && cd /build/Caffe-HRT/unit_tests \
    && make clean \
    && make \
    && /build/gen-pkg-config-pc.sh /usr/local/AID \
    && apt-get install cython \
    && pip install scikit-image protobuf \
    && mkdir /usr/local/AID/opencv \
    && cp -r /build/opencv/build/lib /usr/local/AID/opencv/lib \
    && rm -rf /build

ENV PYTHONPATH=/usr/local/AID/Caffe-HRT/python:/usr/local/AID/opencv/lib