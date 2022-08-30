#include <signal.h>
#include <inttypes.h>

#include "gtest/gtest.h"

#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

namespace demo {

constexpr int kNrParticles = 8192 * 2;
constexpr int kNGrid = 128;
constexpr size_t N_ITER = 50;

class MPM88DemoImpl {
 public:
  MPM88DemoImpl(const std::string &aot_path, TiArch arch) {
    InitTaichiRuntime(arch);

    module_ = ti_load_aot_module(runtime_, aot_path.c_str());

    // Prepare Ndarray for model
    const std::vector<int> shape_1d = {kNrParticles};
    const std::vector<int> shape_2d = {kNGrid, kNGrid};
    const std::vector<int> vec2_shape = {2};
    const std::vector<int> vec3_shape = {3};
    const std::vector<int> mat2_shape = {2, 2};

    x_ = capi::utils::make_ndarray(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                   shape_1d.data(), 1, vec2_shape.data(), 1,
                                   /*host_read=*/false, /*host_write=*/false);

    v_ = capi::utils::make_ndarray(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                   shape_1d.data(), 1, vec2_shape.data(), 1,
                                   /*host_read=*/false, /*host_write=*/false);

    pos_ = capi::utils::make_ndarray(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                     shape_1d.data(), 1, vec3_shape.data(), 1,
                                     /*host_read=*/false, /*host_write=*/false);

    C_ = capi::utils::make_ndarray(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                   shape_1d.data(), 1, mat2_shape.data(), 2,
                                   /*host_read=*/false, /*host_write=*/false);

    J_ = capi::utils::make_ndarray(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                   shape_1d.data(), 1, nullptr, 0,
                                   /*host_read=*/false, /*host_write=*/false);

    grid_v_ =
        capi::utils::make_ndarray(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                  shape_2d.data(), 2, vec2_shape.data(), 1,
                                  /*host_read=*/false, /*host_write=*/false);

    grid_m_ = capi::utils::make_ndarray(
        runtime_, TiDataType::TI_DATA_TYPE_F32, shape_2d.data(), 2, nullptr, 0,
        /*host_read=*/false, /*host_write=*/false);

    k_init_particles_ = ti_get_aot_module_kernel(module_, "init_particles");
    k_substep_g2p_ = ti_get_aot_module_kernel(module_, "substep_g2p");
    k_substep_reset_grid_ =
        ti_get_aot_module_kernel(module_, "substep_reset_grid");
    k_substep_p2g_ = ti_get_aot_module_kernel(module_, "substep_p2g");
    k_substep_update_grid_v_ =
        ti_get_aot_module_kernel(module_, "substep_update_grid_v");

    k_init_particles_args_[0] = x_.arg_;
    k_init_particles_args_[1] = v_.arg_;
    k_init_particles_args_[2] = J_.arg_;

    k_substep_reset_grid_args_[0] = grid_v_.arg_;
    k_substep_reset_grid_args_[1] = grid_m_.arg_;

    k_substep_p2g_args_[0] = x_.arg_;
    k_substep_p2g_args_[1] = v_.arg_;
    k_substep_p2g_args_[2] = C_.arg_;
    k_substep_p2g_args_[3] = J_.arg_;
    k_substep_p2g_args_[4] = grid_v_.arg_;
    k_substep_p2g_args_[5] = grid_m_.arg_;

    k_substep_update_grid_v_args_[0] = grid_v_.arg_;
    k_substep_update_grid_v_args_[1] = grid_m_.arg_;

    k_substep_g2p_args_[0] = x_.arg_;
    k_substep_g2p_args_[1] = v_.arg_;
    k_substep_g2p_args_[2] = C_.arg_;
    k_substep_g2p_args_[3] = J_.arg_;
    k_substep_g2p_args_[4] = grid_v_.arg_;
    k_substep_g2p_args_[5] = pos_.arg_;

    ti_launch_kernel(runtime_, k_init_particles_, 3,
                     &k_init_particles_args_[0]);

    ti_wait(runtime_);
  }

  ~MPM88DemoImpl() {
    ti_destroy_aot_module(module_);
    ti_destroy_runtime(runtime_);
  }

  void Step() {
    for (size_t i = 0; i < N_ITER; i++) {
      ti_launch_kernel(runtime_, k_substep_reset_grid_, 2,
                       &k_substep_reset_grid_args_[0]);
      ti_launch_kernel(runtime_, k_substep_p2g_, 6, &k_substep_p2g_args_[0]);
      ti_launch_kernel(runtime_, k_substep_update_grid_v_, 2,
                       &k_substep_update_grid_v_args_[0]);
      ti_launch_kernel(runtime_, k_substep_g2p_, 6, &k_substep_g2p_args_[0]);
    }
    ti_wait(runtime_);
  }

 private:
  void InitTaichiRuntime(TiArch arch) {
    runtime_ = ti_create_runtime(arch);
  }

  TiRuntime runtime_;
  TiAotModule module_{nullptr};

  capi::utils::TiNdarrayAndMem x_{nullptr};
  capi::utils::TiNdarrayAndMem v_{nullptr};
  capi::utils::TiNdarrayAndMem J_{nullptr};
  capi::utils::TiNdarrayAndMem C_{nullptr};
  capi::utils::TiNdarrayAndMem grid_v_{nullptr};
  capi::utils::TiNdarrayAndMem grid_m_{nullptr};
  capi::utils::TiNdarrayAndMem pos_{nullptr};

  TiKernel k_init_particles_{nullptr};
  TiKernel k_substep_reset_grid_{nullptr};
  TiKernel k_substep_p2g_{nullptr};
  TiKernel k_substep_update_grid_v_{nullptr};
  TiKernel k_substep_g2p_{nullptr};

  TiArgument k_init_particles_args_[3];
  TiArgument k_substep_reset_grid_args_[2];
  TiArgument k_substep_p2g_args_[6];
  TiArgument k_substep_update_grid_v_args_[2];
  TiArgument k_substep_g2p_args_[6];
};

}  // namespace demo

TEST(CapiMpm88Test, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto impl = std::make_unique<demo::MPM88DemoImpl>(aot_mod_ss.str().c_str(),
                                                      TiArch::TI_ARCH_CUDA);
    impl->Step();
  }
}

TEST(CapiMpm88Test, Vulkan) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto impl = std::make_unique<demo::MPM88DemoImpl>(aot_mod_ss.str().c_str(),
                                                      TiArch::TI_ARCH_VULKAN);
    impl->Step();
  }
}

TEST(CapiMpm88Test, Opengl) {
  if (capi::utils::is_opengl_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto impl = std::make_unique<demo::MPM88DemoImpl>(aot_mod_ss.str().c_str(),
                                                      TiArch::TI_ARCH_OPENGL);
    impl->Step();
  }
}
