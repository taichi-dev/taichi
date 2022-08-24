#include "gtest/gtest.h"
#include "taichi/rhi/opengl/opengl_api.h"
#include "tests/cpp/aot/gfx_utils.h"

using namespace taichi;
using namespace lang;

TEST(DeviceTest, GLDevice) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device_ = taichi::lang::opengl::make_opengl_device();

  aot_test_utils::view_devalloc_as_ndarray(device_.get());
}
