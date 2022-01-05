#include "gtest/gtest.h"

#include "taichi/backends/dx/dx_device.h"
#include "taichi/backends/dx/dx_info_queue.h"

namespace taichi {
namespace lang {

TEST(Dx11DeviceCreationTest, CreateDevice) {
  std::unique_ptr<directx11::Dx11Device> device =
      std::make_unique<directx11::Dx11Device>();

  // Should not crash
  EXPECT_TRUE(device != nullptr);
}

}  // namespace lang
}  // namespace taichi
