#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

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

  ti::NdArray<int32_t> arr_array_0 =
      runtime.allocate_ndarray<int32_t>({kArrLen}, {}, true);
  ti::NdArray<int32_t> arr_array_1 =
      runtime.allocate_ndarray<int32_t>({kArrLen}, {1}, true);

  run_graph["base0"] = base0_val;
  run_graph["base1"] = base1_val;
  run_graph["base2"] = base2_val;
  run_graph["arr0"] = arr_array_0;
  run_graph["arr1"] = arr_array_1;
  run_graph.launch();
  runtime.wait();

  // Check Results
  auto *data = reinterpret_cast<int32_t *>(arr_array_0.map());

  for (int i = 0; i < kArrLen; i++) {
    EXPECT_EQ(data[i], 3 * i + base0_val + base1_val + base2_val);
  }

  data = reinterpret_cast<int32_t *>(arr_array_1.map());

  for (int i = 0; i < kArrLen; i++) {
    EXPECT_EQ(data[i], 3 * i + base0_val + base1_val + base2_val);
  }

  arr_array_0.unmap();
  arr_array_1.unmap();
}

void matrix_aot_test(TiArch arch) {
  uint32_t kArrLen = 1;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
  ti::ComputeGraph run_graph = aot_mod.get_compute_graph("run_graph");

  ti::NdArray<int32_t> vec_arr =
      runtime.allocate_ndarray<int32_t>({kArrLen}, {}, true);
  ti::NdArray<int32_t> mat_arr =
      runtime.allocate_ndarray<int32_t>({kArrLen}, {}, true);

  std::vector<int32_t> vec{1, 2, 3};
  std::vector<std::vector<int32_t>> mat{{1, 2}, {3, 4}};

  run_graph["vec"] = vec;
  run_graph["mat"] = mat;
  run_graph["vec_arr"] = vec_arr;
  run_graph["mat_arr"] = mat_arr;
  run_graph.launch();
  runtime.wait();

  auto *data = reinterpret_cast<int32_t *>(vec_arr.map());

  EXPECT_EQ(data[0], 1 + 2 + 3);

  data = reinterpret_cast<int32_t *>(mat_arr.map());

  EXPECT_EQ(data[0], 1 + 2 + 3 + 4);

  vec_arr.unmap();
  mat_arr.unmap();
}

void texture_aot_test(TiArch arch) {
  const uint32_t width = 128;
  const uint32_t height = 128;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str());
  ti::ComputeGraph run_graph = aot_mod.get_compute_graph("run_graph");

  ti::Texture tex0 =
      runtime.allocate_texture2d(width, height, TI_FORMAT_R32F, TI_NULL_HANDLE);
  ti::Texture tex1 =
      runtime.allocate_texture2d(width, height, TI_FORMAT_R32F, TI_NULL_HANDLE);
  ti::NdArray<float> arr =
      runtime.allocate_ndarray<float>({width, height}, {}, true);

  run_graph["tex0"] = tex0;
  run_graph["rw_tex0"] = tex0;
  run_graph["tex1"] = tex1;
  run_graph["rw_tex1"] = tex1;
  run_graph["arr"] = arr;
  run_graph.launch();
  runtime.wait();

  std::vector<float> arr_data(128 * 128);
  arr.read(arr_data);
  for (auto x : arr_data) {
    EXPECT_GT(x, 0.5);
  }
}

TEST_F(CapiTest, GraphTestCpuGraph) {
  TiArch arch = TiArch::TI_ARCH_X64;
  graph_aot_test(arch);
}

TEST_F(CapiTest, GraphTestCudaGraph) {
  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    graph_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestVulkanGraph) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    graph_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestMetalGraph) {
  if (ti::is_arch_available(TI_ARCH_METAL)) {
    TiArch arch = TiArch::TI_ARCH_METAL;
    graph_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestVulkanMatrixGraph) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    matrix_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestOpenglMatrixGraph) {
  if (ti::is_arch_available(TI_ARCH_OPENGL)) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    matrix_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestVulkanTextureGraph) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    texture_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestMetalTextureGraph) {
  if (ti::is_arch_available(TI_ARCH_METAL)) {
    TiArch arch = TiArch::TI_ARCH_METAL;
    texture_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestOpenglGraph) {
  if (ti::is_arch_available(TI_ARCH_OPENGL)) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    graph_aot_test(arch);
  }
}
