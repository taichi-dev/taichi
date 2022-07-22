#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

void graph_aot_test(TiArch arch) {
  uint32_t kArrLen = 100;
  int base0_val = 10;
  int base1_val = 20;
  int base2_val = 30;

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
  TiComputeGraph run_graph =
      ti_get_aot_module_compute_graph(aot_mod, "run_graph");

  // Prepare Arguments
  // base0
  TiArgument base0_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                          .value = {.i32 = base0_val}};
  TiNamedArgument base0_named_arg = {.name = "base0", .argument = base0_arg};

  // base1
  TiArgument base1_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                          .value = {.i32 = base1_val}};
  TiNamedArgument base1_named_arg = {.name = "base1", .argument = base1_arg};

  // base2
  TiArgument base2_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                          .value = {.i32 = base2_val}};
  TiNamedArgument base2_named_arg = {.name = "base2", .argument = base2_arg};

  // arr
  TiMemory arr_memory = ti_allocate_memory(runtime, &alloc_info);
  TiNdArray arr_array = {.memory = arr_memory,
                         .shape = {.dim_count = 1, .dims = {kArrLen}},
                         .elem_shape = {.dim_count = 1, .dims = {1}},
                         .elem_type = TiDataType::TI_DATA_TYPE_I32};
  TiArgumentValue arr_value = {.ndarray = std::move(arr_array)};
  TiArgument arr_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_NDARRAY,
                        .value = std::move(arr_value)};
  TiNamedArgument arr_named_arg = {.name = "arr", .argument = arr_arg};

  // Kernel Execution
  constexpr uint32_t arg_count = 4;
  TiNamedArgument named_args[arg_count] = {
      std::move(base0_named_arg),
      std::move(base1_named_arg),
      std::move(base2_named_arg),
      std::move(arr_named_arg),
  };

  ti_cmd_launch_compute_graph(runtime, run_graph, arg_count, &named_args[0]);

  ti_submit(runtime);
  ti_wait(runtime);

  // Check Results
  auto *data = reinterpret_cast<int32_t *>(ti_map_memory(runtime, arr_memory));

  for (int i = 0; i < kArrLen; i++) {
    EXPECT_EQ(data[i], 3 * i + base0_val + base1_val + base2_val);
  }

  ti_unmap_memory(runtime, arr_memory);
  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

TEST(CapiGraphTest, CpuGraph) {
  TiArch arch = TiArch::TI_ARCH_X64;
  graph_aot_test(arch);
}

TEST(CapiGraphTest, CudaGraph) {
  if (capi::utils::is_cuda_available()) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    graph_aot_test(arch);
  }
}
