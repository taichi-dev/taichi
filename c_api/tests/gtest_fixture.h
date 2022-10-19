#include "gtest/gtest.h"

class CapiTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    auto error_code = ti_get_last_error(0, nullptr);

    if (error_code != TI_ERROR_NOT_SUPPORTED) {
      EXPECT_GE(error_code, TI_ERROR_SUCCESS);
    }
  }
};
