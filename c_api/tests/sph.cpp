#include <signal.h>
#include <inttypes.h>

#include "gtest/gtest.h"

#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"

#define NR_PARTICLES 8000
constexpr int SUBSTEPS = 5;

void run(TiArch arch, const std::string &folder_dir) {
  /* ---------------------------------- */
  /* Runtime & Arguments Initialization */
  /* ---------------------------------- */
  ti::Runtime runtime(arch);

  // Load Aot and Kernel
  ti::AotModule aot_mod = runtime.load_aot_module(folder_dir);

  ti::Kernel k_initialize = aot_mod.get_kernel("initialize");
  ti::Kernel k_initialize_particle = aot_mod.get_kernel("initialize_particle");
  ti::Kernel k_update_density = aot_mod.get_kernel("update_density");
  ti::Kernel k_update_force = aot_mod.get_kernel("update_force");
  ti::Kernel k_advance = aot_mod.get_kernel("advance");
  ti::Kernel k_boundary_handle = aot_mod.get_kernel("boundary_handle");

  const std::vector<uint32_t> shape_1d = {NR_PARTICLES};
  const std::vector<uint32_t> vec3_shape = {3};

  auto N_ = runtime.allocate_ndarray<int32_t>(shape_1d, vec3_shape);
  auto den_ = runtime.allocate_ndarray<float>(shape_1d, {});
  auto pre_ = runtime.allocate_ndarray<float>(shape_1d, {});
  auto pos_ = runtime.allocate_ndarray<float>(shape_1d, vec3_shape);
  auto vel_ = runtime.allocate_ndarray<float>(shape_1d, vec3_shape);
  auto acc_ = runtime.allocate_ndarray<float>(shape_1d, vec3_shape);
  auto boundary_box_ = runtime.allocate_ndarray<float>(shape_1d, vec3_shape);
  auto spawn_box_ = runtime.allocate_ndarray<float>(shape_1d, vec3_shape);
  auto gravity_ = runtime.allocate_ndarray<float>({}, vec3_shape);

  k_initialize[0] = boundary_box_;
  k_initialize[1] = spawn_box_;
  k_initialize[2] = N_;

  k_initialize_particle[0] = pos_;
  k_initialize_particle[1] = spawn_box_;
  k_initialize_particle[2] = N_;
  k_initialize_particle[3] = gravity_;

  k_update_density[0] = pos_;
  k_update_density[1] = den_;
  k_update_density[2] = pre_;

  k_update_force[0] = pos_;
  k_update_force[1] = vel_;
  k_update_force[2] = den_;
  k_update_force[3] = pre_;
  k_update_force[4] = acc_;
  k_update_force[5] = gravity_;

  k_advance[0] = pos_;
  k_advance[1] = vel_;
  k_advance[2] = acc_;

  k_boundary_handle[0] = pos_;
  k_boundary_handle[1] = vel_;
  k_boundary_handle[2] = boundary_box_;

  /* --------------------- */
  /* Kernel Initialization */
  /* --------------------- */
  k_initialize.launch();
  k_initialize_particle.launch();
  runtime.wait();

  /* --------------------- */
  /* Execution & Rendering */
  /* --------------------- */
  for (int i = 0; i < SUBSTEPS; i++) {
    k_update_density.launch();
    k_update_force.launch();
    k_advance.launch();
    k_boundary_handle.launch();
  }
  runtime.wait();
}

TEST(CapiSphTest, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    run(TiArch::TI_ARCH_CUDA, aot_mod_ss.str());
  }
}

TEST(CapiSphTest, Vulkan) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    run(TiArch::TI_ARCH_VULKAN, aot_mod_ss.str());
  }
}

TEST(CapiSphTest, Opengl) {
  if (capi::utils::is_opengl_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    run(TiArch::TI_ARCH_OPENGL, aot_mod_ss.str());
  }
}
