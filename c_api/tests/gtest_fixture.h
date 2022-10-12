#include "gtest/gtest.h"

class CapiTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    auto error_code = ti_get_last_error(0, nullptr);

    if (error_code == TI_ERROR_NOT_SUPPORTED)
      return;
    EXPECT_GE(ti_get_last_error(0, nullptr), TI_ERROR_NOT_SUPPORTED);
  }
};
