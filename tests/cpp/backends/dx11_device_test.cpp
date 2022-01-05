#include "gtest/gtest.h"

#include "taichi/backends/dx/dx_device.h"
#include "taichi/backends/dx/dx_info_queue.h"

namespace taichi {
namespace lang {

TEST(Dx11DeviceCreationTest, CreateDevice) {
  // Enable debug layer
  directx11::debug_enabled(true);

  std::unique_ptr<directx11::Dx11Device> device =
      std::make_unique<directx11::Dx11Device>();

  // Should not crash
  EXPECT_TRUE(device != nullptr);

  // Should have one object of each of the following types:
  // ID3D11Device
  // ID3D11Context
  // ID3DDeviceContextState
  // ID3D11BlendState
  // ID3D11DepthStencilState
  // ID3D11RasterizerState
  // ID3D11Sampler
  // ID3D11Query
  EXPECT_EQ(device->live_dx11_object_count(), 8);
}

TEST(Dx11InfoQueueTest, ParseReferenceCount) {
  const std::vector<std::string> messages = {
      "Create ID3D11Context: Name=\"unnamed\", Addr=0x0000018F6678E080, "
      "ExtRef=1, IntRef=0",
      "Create ID3DDeviceContextState: Name=\"unnamed\", "
      "Addr=0x0000018F6686CE10, "
      "ExtRef=1, IntRef=0",
      "Create ID3D11BlendState: Name=\"unnamed\", Addr=0x0000018F667F6DB0, "
      "ExtRef=1, IntRef=0",
      "Create ID3D11DepthStencilState: Name=\"unnamed\", "
      "Addr=0x0000018F667F6BC0, ExtRef=1, IntRef=0",
      "Create ID3D11RasterizerState: Name=\"unnamed\", "
      "Addr=0x0000018F64891420, "
      "ExtRef=1, IntRef=0",
      "Create ID3D11Sampler: Name=\"unnamed\", Addr=0x0000018F667F6FA0, "
      "ExtRef=1, IntRef=0",
      "Create ID3D11Query: Name=\"unnamed\", Addr=0x0000018F64E81DA0, "
      "ExtRef=1, IntRef=0",
      "Create ID3D11Fence: Name=\"unnamed\", Addr=0x0000018F64FF7380, "
      "ExtRef=1, IntRef=0",
      "Destroy ID3D11Fence: Name=\"unnamed\", Addr=0x0000018F64FF7380",
      "Live ID3D11Device at 0x0000018F66782250, Refcount: 5",
      "Live ID3D11Context at 0x0000018F6678E080, Refcount: 1, IntRef: 1",
      "Live ID3DDeviceContextState at 0x0000018F6686CE10, Refcount: 0, IntRef: "
      "1",
      "Live ID3D11BlendState at 0x0000018F667F6DB0, Refcount: 0, "
      "IntRef: 1",
      "Live ID3D11DepthStencilState at 0x0000018F667F6BC0, Refcount: 0, "
      "IntRef: 1",
      "Live ID3D11RasterizerState at 0x0000018F64891420, Refcount: 0, "
      "IntRef: 1",
      "Live ID3D11Sampler at 0x0000018F667F6FA0, Refcount: 0, IntRef: 1",
      "Live ID3D11Query at 0x0000018F64E81DA0, Refcount: 0, IntRef: 1"};
  std::vector<directx11::Dx11InfoQueue::Entry> entries =
      directx11::Dx11InfoQueue::parse_reference_count(messages);
  EXPECT_EQ(entries.size(), 8);
}

}  // namespace lang
}  // namespace taichi
