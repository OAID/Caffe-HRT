
include ../Makefile.config

CXX_SRCS+=test_pooling_layer.cpp
CXX_SRCS+=test_softmax_layer.cpp
CXX_SRCS+= test_inner_product_layer.cpp
CXX_SRCS+=test_neuron_layer.cpp
CXX_SRCS+=test_lrn_layer.cpp  #failed on single channel LRN
#C_SRCS=pmu.c testbed.c
CXX_SRCS+= test_convolution_layer.cpp  
#CXX_SRCS= test_fail.cpp
BIN_SRCS=test_caffe_main.cpp test.cpp

HOME=/home/firefly
#
#  caffe related stuff
#

CAFFE_ROOT=$(HOME)/CaffeOnACL
CAFFE_INCS = -I$(CAFFE_ROOT)/include -I$(CAFFE_ROOT)/distribute/include/
CAFFE_LIBS = -L$(CAFFE_ROOT)/distribute/lib -lcaffe  -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem
CAFFE_RPATH =$(CAFFE_ROOT)/distribute/lib


#
# google test related stuff
#

GTEST_ROOT=/usr/local
GTEST_LIBS=$(GTEST_ROOT)/lib/libgtest_main.a $(GTEST_ROOT)/lib/libgtest.a
GTEST_INCS=-I$(GTEST_ROOT)/include/


#
# arm compute
#

ACL_ROOT=$(HOME)/ComputeLibrary
ACL_LIBS=-L$(ACL_ROOT)/build -L$(ACL_ROOT)/build/arm_compute -larm_compute  -lOpenCL
ACL_RPATH=$(ACL_ROOT)/build:$(ACL_ROOT)/build/arm_compute

COMM_CFLAGS=$(GTEST_INCS) $(CAFFE_INCS) -Wall -g  -DCPU_ONLY -DUSE_ACL 
#USE_PROFILING -- get profiling informations, is controled by LOGACL
#LAYER_PERF_STAT -- haitao's net profiling information
ifeq ($(USE_PROFILING), 1)
        COMM_CFLAGS += -DUSE_PROFILING -DLAYER_PERF_STAT
endif

CXXFLAGS=$(COMM_CFLAGS) -Wno-sign-compare
CFLAGS=$(COMM_CFLAGS)

CC=gcc
CXX=g++

COMM_OBJS=$(CXX_SRCS:.cpp=.o) $(C_SRCS:.c=.o)
BIN_OBJS+=$(BIN_SRCS:.cpp=.o)
BIN_EXES=$(BIN_OBJS:.o=)

LIBS+=$(CAFFE_LIBS) $(GTEST_LIBS) -lpthread -lopenblas $(ACL_LIBS)

RT_PATH=-Wl,-rpath,$(CAFFE_RPATH):$(ACL_RPATH)

LDFLAGS+=$(RT_PATH)


all : $(BIN_EXES)

$(BIN_EXES):%:%.o

$(BIN_EXES):$(COMM_OBJS)



clean:
	rm -f $(BIN_EXES) *.o *.so

.PHONY : all clean

%.o : %.c
	$(CC) $(CFLAGS)  -c $< -o $@ 

%.o : %.cpp
	$(CXX) $(CXXFLAGS)  -c $< -o $@ 

%: %.o
	$(CXX) $(LDFLAGS) $< $(COMM_OBJS) -o $@ $(LIBS)

