CC = gcc
CXX = g++

ifeq ($(ACL_ROOT),)
$(error ACL_ROOT does not specified. use "export ACL_ROOT='path of acl soure directory'")
endif

ifeq ($(CAFFE_ROOT),)
$(error CAFFE_ROOT does not specified. use "export CAFFE_ROOT='path of caffe soure directory'")
endif

include $(CAFFE_ROOT)/Makefile.config

#
#  caffe related stuff
#
CAFFE_INCS = -I$(CAFFE_ROOT)/include -I$(CAFFE_ROOT)/distribute/include/
CAFFE_LIBS = -L$(CAFFE_ROOT)/distribute/lib -lcaffe  -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem

#
# google test related stuff
#
GTEST_ROOT = /usr/local
GTEST_INCS = -I$(GTEST_ROOT)/include/
GTEST_LIBS = -L$(GTEST_ROOT)/lib/ -lgtest_main -lgtest

#
# arm compute
#
ACL_LIBS  = -L$(ACL_ROOT)/build -L$(ACL_ROOT)/build/arm_compute -larm_compute  -lOpenCL

COMM_CFLAGS = -Wall -g -DCPU_ONLY -DUSE_ACL $(GTEST_INCS) $(CAFFE_INCS)

#USE_PROFILING -- get profiling informations, is controled by LOGACL
#LAYER_PERF_STAT -- haitao's net profiling information
ifeq ($(USE_PROFILING), 1)
        COMM_CFLAGS += -DUSE_PROFILING -DLAYER_PERF_STAT
endif

CXXFLAGS = $(COMM_CFLAGS) -Wno-sign-compare
CFLAGS   = $(COMM_CFLAGS)

LDFLAGS += $(CAFFE_LIBS) $(GTEST_LIBS) $(ACL_LIBS) -lpthread -lopenblas

COMM_OBJS += test_pooling_layer.o
COMM_OBJS += test_softmax_layer.o
COMM_OBJS += test_inner_product_layer.o
COMM_OBJS += test_neuron_layer.o
COMM_OBJS += test_lrn_layer.o
COMM_OBJS += test_convolution_layer.o

BIN_EXES = test test_caffe_main

all : $(BIN_EXES)

test : test.o $(COMM_OBJS)
	$(CXX) $< $(COMM_OBJS) -o $@ $(LDFLAGS)

test_caffe_main : test_caffe_main.o $(COMM_OBJS)
	$(CXX) $< $(COMM_OBJS) -o $@ $(LDFLAGS)

clean:
	$(RM) $(BIN_EXES) *.o *.so

.PHONY : all clean

%.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@ 

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@ 
