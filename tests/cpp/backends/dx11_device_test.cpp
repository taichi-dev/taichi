#include "gtest/gtest.h"

#ifdef TI_WITH_DX11

#include "taichi/ir/ir_builder.h"
#include "taichi/backends/dx/dx_device.h"
#include "taichi/backends/dx/dx_info_queue.h"
#include "taichi/backends/dx/dx_program.h"
#include "taichi/system/memory_pool.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {
namespace directx11 {

TEST(Dx11DeviceCreationTest, CreateDeviceAndAllocateMemory) {
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
  int count0, count1, count2;
  if (kD3d11DebugEnabled) {
    count0 = device->live_dx11_object_count();
    EXPECT_EQ(count0, 8);
  }

  taichi::lang::Device::AllocParams params;
  params.size = 1048576;
  const taichi::lang::DeviceAllocation device_alloc =
      device->allocate_memory(params);

  // The purpose of the device_alloc_guard is to rule out double free
  const taichi::lang::DeviceAllocationGuard device_alloc_guard(device_alloc);
  if (kD3d11DebugEnabled) {
    count1 = device->live_dx11_object_count();
    // Should have allocated an UAV and a Buffer, so 2 more objects.
    EXPECT_EQ(count1 - count0, 2);
  }

  // Map to CPU, write some values, then check those values
  void *mapped = device->map(device_alloc);
  int *mapped_int = reinterpret_cast<int *>(mapped);
  for (int i = 0; i < 100; i++) {
    mapped_int[i] = i;
  }
  device->unmap(device_alloc);

  mapped = device->map(device_alloc);
  mapped_int = reinterpret_cast<int *>(mapped);
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(mapped_int[i], i);
  }
  device->unmap(device_alloc);

  // The 2 objects should have been released.
  device->dealloc_memory(device_alloc);
  if (kD3d11DebugEnabled) {
    count2 = device->live_dx11_object_count();
    EXPECT_EQ(count2 - count1, -2);
  }
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

TEST(Dx11StreamTest, CommandListTest) {
  std::unique_ptr<directx11::Dx11Device> device =
      std::make_unique<directx11::Dx11Device>();
  std::unique_ptr<Dx11Stream> stream =
      std::make_unique<Dx11Stream>(device.get());
  stream->new_command_list();
}

TEST(Dx11ProgramTest, MaterializeRuntimeTest) {
  std::unique_ptr<directx11::Dx11Device> device =
      std::make_unique<directx11::Dx11Device>();
  std::unique_ptr<MemoryPool> pool =
      std::make_unique<MemoryPool>(Arch::dx11, device.get());
  std::unique_ptr<Dx11ProgramImpl> program =
      std::make_unique<Dx11ProgramImpl>(default_compile_config);
  /*
  This test needs allocate_memory because of the call stack here:
  Dx11ProgramImpl::materialize_runtime
  - GfxRuntime::GfxRuntime
     - GfxRuntime::init_buffers
        - Dx11Device::allocate_memory_unique
        - Dx11Device::get_compute_stream
        - Dx11Stream::new_command_list
        - Dx11Stream::buffer_fill
        - Dx11Stream::submit_synced
  */
  uint64_t *result_buffer;
  program->materialize_runtime(pool.get(), nullptr, &result_buffer);

  TestProgram test_prog;
  test_prog.setup();

  IRBuilder builder;
  auto *lhs = builder.get_int32(42);

  auto block = builder.extract_ir();
  test_prog.prog()->config.arch = Arch::dx11;
  auto ker = std::make_unique<Kernel>(*test_prog.prog(), std::move(block));
  program->compile(ker.get(), nullptr);
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi

#endif
