#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

static void kernel_aot_test(TiArch arch) {
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
  constexpr uint32_t arg_count = 2;
  TiArgument args[arg_count] = {std::move(arg0), std::move(arg1)};

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

static void field_aot_test(TiArch arch) {
  int base_val = 10;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  TiRuntime runtime = ti_create_runtime(arch);

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, aot_mod_ss.str().c_str());

  TiKernel k_init_fields = ti_get_aot_module_kernel(aot_mod, "init_fields");
  TiKernel k_check_init_x = ti_get_aot_module_kernel(aot_mod, "check_init_x");
  TiKernel k_check_init_y = ti_get_aot_module_kernel(aot_mod, "check_init_y");
  TiKernel k_deactivate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "deactivate_pointer_fields");
  TiKernel k_activate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "activate_pointer_fields");
  TiKernel k_check_deactivate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "check_deactivate_pointer_fields");
  TiKernel k_check_activate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "check_activate_pointer_fields");

  // Prepare Arguments
  TiArgument base_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                         .value = {.i32 = base_val}};

  constexpr uint32_t arg_count = 1;
  TiArgument args[arg_count] = {std::move(base_arg)};

  // Kernel Execution
  ti_launch_kernel(runtime, k_init_fields, arg_count, &args[0]);
  ti_launch_kernel(runtime, k_check_init_x, arg_count, &args[0]);
  ti_launch_kernel(runtime, k_check_init_y, 0 /*arg_count*/, &args[0]);

  ti_launch_kernel(runtime, k_deactivate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);
  ti_launch_kernel(runtime, k_check_deactivate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);
  ti_launch_kernel(runtime, k_activate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);
  ti_launch_kernel(runtime, k_check_activate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);

  // Check Results
  capi::utils::check_runtime_error(runtime);

  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

TEST(CapiAotTest, CpuField) {
  TiArch arch = TiArch::TI_ARCH_X64;
  field_aot_test(arch);
}

TEST(CapiAotTest, CudaField) {
  if (capi::utils::is_cuda_available()) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    field_aot_test(arch);
  }
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
