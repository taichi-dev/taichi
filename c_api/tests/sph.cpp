#include <signal.h>
#include <inttypes.h>

#include "gtest/gtest.h"

#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

#define NR_PARTICLES 8000
constexpr int SUBSTEPS = 5;

void run(TiArch arch, const std::string &folder_dir) {
  /* ---------------------------------- */
  /* Runtime & Arguments Initialization */
  /* ---------------------------------- */
  TiRuntime runtime = ti_create_runtime(arch);

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, folder_dir.c_str());

  TiKernel k_initialize = ti_get_aot_module_kernel(aot_mod, "initialize");
  TiKernel k_initialize_particle =
      ti_get_aot_module_kernel(aot_mod, "initialize_particle");
  TiKernel k_update_density =
      ti_get_aot_module_kernel(aot_mod, "update_density");
  TiKernel k_update_force = ti_get_aot_module_kernel(aot_mod, "update_force");
  TiKernel k_advance = ti_get_aot_module_kernel(aot_mod, "advance");
  TiKernel k_boundary_handle =
      ti_get_aot_module_kernel(aot_mod, "boundary_handle");

  const std::vector<int> shape_1d = {NR_PARTICLES};
  const std::vector<int> vec3_shape = {3};

  auto N_ = capi::utils::make_ndarray(runtime, TiDataType::TI_DATA_TYPE_I32,
                                      shape_1d.data(), 1, vec3_shape.data(), 1,
                                      false /*host_read*/, false /*host_write*/
  );
  auto den_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1, nullptr, 0,
      false /*host_read*/, false /*host_write*/
  );
  auto pre_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1, nullptr, 0,
      false /*host_read*/, false /*host_write*/
  );

  auto pos_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1,
      vec3_shape.data(), 1, false /*host_read*/, false /*host_write*/
  );
  auto vel_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1,
      vec3_shape.data(), 1, false /*host_read*/, false /*host_write*/
  );
  auto acc_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1,
      vec3_shape.data(), 1, false /*host_read*/, false /*host_write*/
  );
  auto boundary_box_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1,
      vec3_shape.data(), 1, false /*host_read*/, false /*host_write*/
  );
  auto spawn_box_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, shape_1d.data(), 1,
      vec3_shape.data(), 1, false /*host_read*/, false /*host_write*/
  );
  auto gravity_ = capi::utils::make_ndarray(
      runtime, TiDataType::TI_DATA_TYPE_F32, nullptr, 0, vec3_shape.data(), 1,
      false /*host_read*/, false /*host_write*/);

  TiArgument k_initialize_args[3];
  TiArgument k_initialize_particle_args[4];
  TiArgument k_update_density_args[3];
  TiArgument k_update_force_args[6];
  TiArgument k_advance_args[3];
  TiArgument k_boundary_handle_args[3];

  k_initialize_args[0] = boundary_box_.arg_;
  k_initialize_args[1] = spawn_box_.arg_;
  k_initialize_args[2] = N_.arg_;

  k_initialize_particle_args[0] = pos_.arg_;
  k_initialize_particle_args[1] = spawn_box_.arg_;
  k_initialize_particle_args[2] = N_.arg_;
  k_initialize_particle_args[3] = gravity_.arg_;

  k_update_density_args[0] = pos_.arg_;
  k_update_density_args[1] = den_.arg_;
  k_update_density_args[2] = pre_.arg_;

  k_update_force_args[0] = pos_.arg_;
  k_update_force_args[1] = vel_.arg_;
  k_update_force_args[2] = den_.arg_;
  k_update_force_args[3] = pre_.arg_;
  k_update_force_args[4] = acc_.arg_;
  k_update_force_args[5] = gravity_.arg_;

  k_advance_args[0] = pos_.arg_;
  k_advance_args[1] = vel_.arg_;
  k_advance_args[2] = acc_.arg_;

  k_boundary_handle_args[0] = pos_.arg_;
  k_boundary_handle_args[1] = vel_.arg_;
  k_boundary_handle_args[2] = boundary_box_.arg_;

  /* --------------------- */
  /* Kernel Initialization */
  /* --------------------- */
  ti_launch_kernel(runtime, k_initialize, 3, &k_initialize_args[0]);
  ti_launch_kernel(runtime, k_initialize_particle, 4,
                   &k_initialize_particle_args[0]);
  ti_wait(runtime);

  /* --------------------- */
  /* Execution & Rendering */
  /* --------------------- */
  for (int i = 0; i < SUBSTEPS; i++) {
    ti_launch_kernel(runtime, k_update_density, 3, &k_update_density_args[0]);
    ti_launch_kernel(runtime, k_update_force, 6, &k_update_force_args[0]);
    ti_launch_kernel(runtime, k_advance, 3, &k_advance_args[0]);
    ti_launch_kernel(runtime, k_boundary_handle, 3, &k_boundary_handle_args[0]);
  }
  ti_wait(runtime);
}

TEST(CapiSphTest, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    run(TiArch::TI_ARCH_CUDA, aot_mod_ss.str().c_str());
  }
}

TEST(CapiSphTest, Vulkan) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    run(TiArch::TI_ARCH_VULKAN, aot_mod_ss.str().c_str());
  }
}

TEST(CapiSphTest, Opengl) {
  if (capi::utils::is_opengl_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    run(TiArch::TI_ARCH_OPENGL, aot_mod_ss.str().c_str());
  }
}
