#include "gtest/gtest.h"

inline bool is_error_ignorable(TiError error) {
  if (error == TI_ERROR_NOT_SUPPORTED)
    return true;

  return false;
}

class CapiTest : public ::testing::Test {
 public:
  void ASSERT_TAICHI_SUCCESS() {
    ti::Error actual = ti::get_last_error();
    EXPECT_EQ(actual.error, TI_ERROR_SUCCESS);
  }

  void EXPECT_TAICHI_ERROR(TiError expected,
                           const std::string &match = "",
                           bool reset_error = true) {
    ti::Error err = ti::get_last_error();

    EXPECT_EQ(err.error, expected);

    if (!match.empty())
      EXPECT_NE(err.message.find(match), std::string::npos);

    if (reset_error)
      ti::set_last_error(TI_ERROR_SUCCESS);
  }

 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    ti::Error err = ti::get_last_error();

    if (!is_error_ignorable(err.error)) {
      EXPECT_GE(err.error, TI_ERROR_SUCCESS);
    }
  }
};
