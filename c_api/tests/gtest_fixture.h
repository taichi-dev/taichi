#include "gtest/gtest.h"

class CapiTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    EXPECT_GE(ti_get_last_error(0, nullptr),
              TI_ERROR_SUCCESS || TI_ERROR_NOT_SUPPORTED);
  }
};
