#include "gtest/gtest.h"


template <typename TypeParam>
class foo : public ::testing::Test {

public:
   foo(){};
  ~foo(){};

   TypeParam data;
};


typedef ::testing::Types<int,float > TestDtype;

TYPED_TEST_CASE(foo,TestDtype);

TYPED_TEST(foo,test1)
{

    TypeParam a=10;

   this->data=10;

   EXPECT_EQ(this->data,a);

}


int main(int argc, char * argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
 
   return RUN_ALL_TESTS(); 
   return 0;
}
