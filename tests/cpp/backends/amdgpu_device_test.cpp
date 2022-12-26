#include "gtest/gtest.h"

#ifdef TI_WITH_AMDGPU
#include "taichi/ir/ir_builder.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/amdgpu/amdgpu_device.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {
TEST(AMDGPU, CreateDeviceAndAlloc) {
  std::unique_ptr<amdgpu::AmdgpuDevice> device =
      std::make_unique<amdgpu::AmdgpuDevice>();
  EXPECT_TRUE(device != nullptr);
  taichi::lang::Device::AllocParams params;
  params.size = 400;
  params.host_read = true;
  params.host_write = true;
  const taichi::lang::DeviceAllocation device_alloc =
      device->allocate_memory(params);

  // The purpose of the device_alloc_guard is to rule out double free
  const taichi::lang::DeviceAllocationGuard device_alloc_guard(device_alloc);
  // Map to CPU, write some values, then check those values
  void *mapped;
  EXPECT_EQ(device->map(device_alloc, &mapped), RhiResult::success);

  int *mapped_int = reinterpret_cast<int *>(mapped);
  for (int i = 0; i < params.size / sizeof(int); i++) {
    mapped_int[i] = i;
  }
  device->unmap(device_alloc);
  EXPECT_EQ(device->map(device_alloc, &mapped), RhiResult::success);

  mapped_int = reinterpret_cast<int *>(mapped);
  for (int i = 0; i < params.size / sizeof(int); i++) {
    EXPECT_EQ(mapped_int[i], i);
  }
  device->unmap(device_alloc);
}

TEST(AMDGPU, ImportMemory) {
  std::unique_ptr<amdgpu::AmdgpuDevice> device =
      std::make_unique<amdgpu::AmdgpuDevice>();
  EXPECT_TRUE(device != nullptr);

  int *ptr = nullptr;
  size_t mem_size = 400;
  AMDGPUDriver::get_instance().malloc_managed((void **)&ptr, mem_size,
                                              HIP_MEM_ATTACH_GLOBAL);
  const taichi::lang::DeviceAllocation device_alloc =
      device->import_memory(ptr, mem_size);

  for (int i = 0; i < mem_size / sizeof(int); i++) {
    ptr[i] = i;
  }

  taichi::lang::Device::AllocParams params;
  params.size = 400;
  params.host_read = true;
  params.host_write = true;
  const taichi::lang::DeviceAllocation device_dest =
      device->allocate_memory(params);
  const taichi::lang::DeviceAllocationGuard device_dest_guard(device_dest);

  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  device->memcpy_internal(device_dest.get_ptr(0), device_alloc.get_ptr(0), params.size);
  void *mapped;
  EXPECT_EQ(device->map(device_dest, &mapped), RhiResult::success);

  int *mapped_int = reinterpret_cast<int *>(mapped);

  for (int i = 0; i < params.size / sizeof(int); i++) {
    EXPECT_EQ(mapped_int[i], i);
  }
  device->unmap(device_dest);
  // import memory should been deallocated manually
  AMDGPUDriver::get_instance().mem_free(ptr);
}

TEST(AMDGPU, CreateContextAndGetMemInfo) {
  auto total_size = AMDGPUContext::get_instance().get_total_memory();
  auto free_size = AMDGPUContext::get_instance().get_free_memory();
  EXPECT_GE(total_size, free_size);
  EXPECT_GE(free_size, 0);
}

}  // namespace lang
}  // namespace taichi
#endif
