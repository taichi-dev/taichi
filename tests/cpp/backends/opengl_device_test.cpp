#include "gtest/gtest.h"

#ifdef TI_WITH_OPENGL

#include "taichi/rhi/opengl/opengl_api.h"
#include "taichi/runtime/program_impls/opengl/opengl_program.h"

#include "tests/cpp/backends/device_test_utils.h"
#include "taichi/system/memory_pool.h"

namespace taichi::lang {

TEST(GLDeviceTest, CreateDeviceAndAllocateMemory) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device_ = taichi::lang::opengl::make_opengl_device();

  // Run memory allocation tests
  device_test_utils::test_memory_allocation(device_.get());
  device_test_utils::test_view_devalloc_as_ndarray(device_.get());
}

TEST(GLDeviceTest, MaterializeRuntimeTest) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device_ = taichi::lang::opengl::make_opengl_device();

  std::unique_ptr<MemoryPool> pool =
      std::make_unique<MemoryPool>(Arch::opengl, device_.get());
  std::unique_ptr<OpenglProgramImpl> program =
      std::make_unique<OpenglProgramImpl>(default_compile_config);
  uint64_t *result_buffer;
  program->materialize_runtime(pool.get(), nullptr, &result_buffer);

  device_test_utils::test_program(program.get(), Arch::opengl);
}

}  // namespace taichi::lang

#endif
