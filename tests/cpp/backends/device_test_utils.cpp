#include "taichi/program/ndarray.h"
#include "taichi/system/memory_pool.h"
#include "taichi/ir/ir_builder.h"

#include "tests/cpp/backends/device_test_utils.h"
#include "tests/cpp/program/test_program.h"

namespace device_test_utils {

using namespace taichi::lang;

void test_memory_allocation(taichi::lang::Device *device) {
  taichi::lang::Device::AllocParams params;
  params.size = 1048576;
  params.host_read = true;
  params.host_write = true;
  const taichi::lang::DeviceAllocation device_alloc =
      device->allocate_memory(params);

  // The purpose of the device_alloc_guard is to rule out double free
  const taichi::lang::DeviceAllocationGuard device_alloc_guard(device_alloc);

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
}

void test_view_devalloc_as_ndarray(taichi::lang::Device *device_) {
  const int size = 40;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  taichi::lang::DeviceAllocation devalloc_arr_ = device_->allocate_memory(alloc_params);

  std::vector<int> element_shape = {4};
  auto arr1 = taichi::lang::Ndarray(devalloc_arr_, taichi::lang::PrimitiveType::i32, {10}, element_shape);
  EXPECT_TRUE(arr1.get_element_shape() == element_shape);
  EXPECT_EQ(arr1.total_shape()[0], 10);
  EXPECT_EQ(arr1.total_shape()[1], 4);

  auto arr2 = taichi::lang::Ndarray(devalloc_arr_, taichi::lang::PrimitiveType::i32, {10}, element_shape,
                      ExternalArrayLayout::kSOA);
  EXPECT_TRUE(arr2.get_element_shape() == element_shape);
  EXPECT_EQ(arr2.total_shape()[0], 4);
  EXPECT_EQ(arr2.total_shape()[1], 10);

  device_->dealloc_memory(devalloc_arr_);
}

void test_program(taichi::lang::ProgramImpl* program, taichi::Arch arch) {
  TestProgram test_prog;
  test_prog.setup();

  IRBuilder builder;
  auto block = builder.extract_ir();
  test_prog.prog()->this_thread_config().arch = arch;
  auto ker = std::make_unique<taichi::lang::Kernel>(*test_prog.prog(), std::move(block));
  program->compile(ker.get(), nullptr);
}
};  // namespace device_test_utils
