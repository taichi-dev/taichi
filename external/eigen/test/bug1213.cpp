
// This anonymous enum is essential to trigger the linking issue
enum {
  Foo
};

#include "bug1213.h"

bool bug1213_1(const Eigen::Vector3f& x)
{
  return bug1213_2(x);
}

