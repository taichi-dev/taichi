#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

void kernel_aot_test(TiArch arch) {
  uint32_t kArrLen = 32;
  int arg0_val = 0;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  TiMemoryAllocateInfo alloc_info;
  alloc_info.size = kArrLen * sizeof(int32_t);
  alloc_info.host_write = false;
  alloc_info.host_read = false;
  alloc_info.export_sharing = false;
  alloc_info.usage = TiMemoryUsageFlagBits::TI_MEMORY_USAGE_STORAGE_BIT;

  TiRuntime runtime = ti_create_runtime(arch);

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, aot_mod_ss.str().c_str());
  TiKernel k_run = ti_get_aot_module_kernel(aot_mod, "run");

  // Prepare Arguments
  TiArgument arg0 = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                     .value = {.i32 = arg0_val}};

  TiMemory memory = ti_allocate_memory(runtime, &alloc_info);
  TiNdArray arg_array = {.memory = memory,
                         .shape = {.dim_count = 1, .dims = {kArrLen}},
                         .elem_shape = {.dim_count = 1, .dims = {1}},
                         .elem_type = TiDataType::TI_DATA_TYPE_I32};

  TiArgumentValue arg_value = {.ndarray = std::move(arg_array)};

  TiArgument arg1 = {.type = TiArgumentType::TI_ARGUMENT_TYPE_NDARRAY,
                     .value = std::move(arg_value)};

  // Kernel Execution
  uint32_t arg_count = 2;
  TiArgument args[2] = {std::move(arg0), std::move(arg1)};

  ti_launch_kernel(runtime, k_run, arg_count, &args[0]);

  // Check Results
  auto *data = reinterpret_cast<int32_t *>(ti_map_memory(runtime, memory));

  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], i);
  }

  ti_unmap_memory(runtime, memory);
  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

TEST(CapiAotTest, CpuKernel) {
  TiArch arch = TiArch::TI_ARCH_X64;
  kernel_aot_test(arch);
}

TEST(CapiAotTest, CudaKernel) {
  if (capi::utils::is_cuda_available()) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    kernel_aot_test(arch);
  }
}
