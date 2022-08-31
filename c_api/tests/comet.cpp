#include <signal.h>
#include <inttypes.h>

#include "gtest/gtest.h"

#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"

constexpr int img_h = 680;
constexpr int img_w = 680;
constexpr int img_c = 4;

static void comet_run(TiArch arch, const std::string &folder_dir) {
  ti::Runtime runtime(arch);

  // Load Aot and Kernel
  ti::AotModule aot_mod = runtime.load_aot_module(folder_dir.c_str());

  ti::ComputeGraph g_init = aot_mod.get_compute_graph("init");
  ti::ComputeGraph g_update = aot_mod.get_compute_graph("update");

  ti::NdArray<float> arg_array =
      runtime.allocate_ndarray<float>({img_h, img_w, img_c}, {});

  g_init["arr"] = arg_array;
  g_init.launch();

  runtime.submit();
  runtime.wait();
  for (int i = 0; i < 10000; i++) {
    g_update["arg"] = arg_array;
    runtime.wait();
  }
}

TEST(CapiCometTest, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    comet_run(TiArch::TI_ARCH_CUDA, aot_mod_ss.str());
  }
}
