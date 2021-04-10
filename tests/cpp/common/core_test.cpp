#include "gtest/gtest.h"

#include "taichi/common/core.h"

namespace taichi {

TEST(CoreTest, Basic) {
  EXPECT_EQ(trim_string("hello taichi  "), "hello taichi");
}

}  // namespace taichi
