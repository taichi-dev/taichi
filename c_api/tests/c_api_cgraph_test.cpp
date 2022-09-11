#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"

void graph_aot_test(TiArch arch) {
  uint32_t kArrLen = 100;
  int base0_val = 10;
  int base1_val = 20;
  int base2_val = 30;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
  ti::ComputeGraph run_graph = aot_mod.get_compute_graph("run_graph");

  ti::NdArray<int32_t> arr_array =
      runtime.allocate_ndarray<int32_t>({kArrLen}, {1}, true);

  run_graph["base0"] = base0_val;
  run_graph["base1"] = base1_val;
  run_graph["base2"] = base2_val;
  run_graph["arr"] = arr_array;
  run_graph.launch();
  runtime.wait();

  // Check Results
  auto *data = reinterpret_cast<int32_t *>(arr_array.map());

  for (int i = 0; i < kArrLen; i++) {
    EXPECT_EQ(data[i], 3 * i + base0_val + base1_val + base2_val);
  }
  arr_array.unmap();
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

TEST(CapiGraphTest, VulkanGraph) {
  if (capi::utils::is_vulkan_available()) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    graph_aot_test(arch);
  }
}

TEST(CapiGraphTest, OpenglGraph) {
  if (capi::utils::is_opengl_available()) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    graph_aot_test(arch);
  }
}
