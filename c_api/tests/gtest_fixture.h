#include "gtest/gtest.h"

inline bool is_error_ignorable(TiError error) {
  if (error == TI_ERROR_NOT_SUPPORTED)
    return true;

  return false;
}

class CapiTest : public ::testing::Test {
 public:
  void ASSERT_TAICHI_SUCCESS() {
    TiError actual = ti_get_last_error(0, nullptr);
    EXPECT_EQ(actual, TI_ERROR_SUCCESS);
  }

  void EXPECT_TAICHI_ERROR(TiError expected,
                           const std::string &match = "",
                           bool reset_error = true) {
    char err_msg[4096]{0};
    TiError err = ti_get_last_error(sizeof(err_msg), err_msg);

    EXPECT_EQ(err, expected);

    if (!match.empty())
      EXPECT_NE(std::string(err_msg).find(match), std::string::npos);

    if (reset_error)
      ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  }

 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    auto error_code = ti_get_last_error(0, nullptr);

    if (!is_error_ignorable(error_code)) {
      EXPECT_GE(error_code, TI_ERROR_SUCCESS);
    }
  }
};
