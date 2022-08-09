#include <signal.h>
#include <inttypes.h>

#include "gtest/gtest.h"

#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

constexpr int img_h = 680;
constexpr int img_w = 680;
constexpr int img_c = 4;

static void comet_run(TiArch arch, const std::string &folder_dir) {
  TiRuntime runtime = ti_create_runtime(arch);

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, folder_dir.c_str());

  TiComputeGraph g_init = ti_get_aot_module_compute_graph(aot_mod, "init");
  TiComputeGraph g_update = ti_get_aot_module_compute_graph(aot_mod, "update");

  // k_img_to_ndarray(args)
  TiMemoryAllocateInfo alloc_info;
  alloc_info.size = img_h * img_w * img_c * sizeof(float);
  alloc_info.host_write = false;
  alloc_info.host_read = false;
  alloc_info.export_sharing = false;
  alloc_info.usage = TiMemoryUsageFlagBits::TI_MEMORY_USAGE_STORAGE_BIT;

  TiMemory memory = ti_allocate_memory(runtime, &alloc_info);
  TiNdArray arg_array = {
      .memory = memory,
      .shape = {.dim_count = 3, .dims = {img_h, img_w, img_c}},
      .elem_shape = {.dim_count = 0, .dims = {0}},
      .elem_type = TiDataType::TI_DATA_TYPE_F32};

  TiArgumentValue arg_value = {.ndarray = std::move(arg_array)};

  TiArgument arr_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_NDARRAY,
                        .value = std::move(arg_value)};
  TiNamedArgument arr_named_arg = {.name = "arr",
                                   .argument = std::move(arr_arg)};
  ;
  TiNamedArgument args[1] = {std::move(arr_named_arg)};

  ti_launch_compute_graph(runtime, g_init, 0, &args[0]);
  ti_wait(runtime);
  for (int i = 0; i < 10000; i++) {
    ti_launch_compute_graph(runtime, g_update, 1, &args[0]);
    ti_wait(runtime);
  }

  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

TEST(CapiCometTest, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    comet_run(TiArch::TI_ARCH_CUDA, aot_mod_ss.str());
  }
}
