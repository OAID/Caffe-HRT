#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

extern "C" {
#include "testbed.h"
}

class testbed_env: public ::testing::Environment {

  public:
      testbed_env(){};
      ~testbed_env() {};

    void SetUp(void) 
    { 
         std::cout<<"setting up testbed resource"<<std::endl;
    }

    void TearDown(void) 
    { 
        std::cout<<"release testbed resource"<<std::endl;
    }

};


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new testbed_env);
  // invoke the test.
  return RUN_ALL_TESTS();
}
