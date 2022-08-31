#include <signal.h>
#include <inttypes.h>

#include "gtest/gtest.h"

#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"

namespace demo {

constexpr int kNrParticles = 8192 * 2;
constexpr int kNGrid = 128;
constexpr size_t N_ITER = 50;

class MPM88DemoImpl {
 public:
  MPM88DemoImpl(const std::string &aot_path, TiArch arch) {
    runtime_ = ti::Runtime(arch);

    module_ = runtime_.load_aot_module(aot_path.c_str());

    // Prepare Ndarray for model
    const std::vector<uint32_t> shape_1d = {kNrParticles};
    const std::vector<uint32_t> shape_2d = {kNGrid, kNGrid};
    const std::vector<uint32_t> vec2_shape = {2};
    const std::vector<uint32_t> vec3_shape = {3};
    const std::vector<uint32_t> mat2_shape = {2, 2};

    x_ = runtime_.allocate_ndarray<float>(shape_1d, vec2_shape);
    v_ = runtime_.allocate_ndarray<float>(shape_1d, vec2_shape);
    pos_ = runtime_.allocate_ndarray<float>(shape_1d, vec3_shape);
    C_ = runtime_.allocate_ndarray<float>(shape_1d, mat2_shape);
    J_ = runtime_.allocate_ndarray<float>(shape_1d, {});
    grid_v_ = runtime_.allocate_ndarray<float>(shape_2d, vec2_shape);
    grid_m_ = runtime_.allocate_ndarray<float>(shape_2d, {});

    k_init_particles_ = module_.get_kernel("init_particles");
    k_substep_g2p_ = module_.get_kernel("substep_g2p");
    k_substep_reset_grid_ = module_.get_kernel("substep_reset_grid");
    k_substep_p2g_ = module_.get_kernel("substep_p2g");
    k_substep_update_grid_v_ = module_.get_kernel("substep_update_grid_v");

    k_init_particles_[0] = x_;
    k_init_particles_[1] = v_;
    k_init_particles_[2] = J_;

    k_substep_reset_grid_[0] = grid_v_;
    k_substep_reset_grid_[1] = grid_m_;

    k_substep_p2g_[0] = x_;
    k_substep_p2g_[1] = v_;
    k_substep_p2g_[2] = C_;
    k_substep_p2g_[3] = J_;
    k_substep_p2g_[4] = grid_v_;
    k_substep_p2g_[5] = grid_m_;

    k_substep_update_grid_v_[0] = grid_v_;
    k_substep_update_grid_v_[1] = grid_m_;

    k_substep_g2p_[0] = x_;
    k_substep_g2p_[1] = v_;
    k_substep_g2p_[2] = C_;
    k_substep_g2p_[3] = J_;
    k_substep_g2p_[4] = grid_v_;
    k_substep_g2p_[5] = pos_;

    k_init_particles_.launch();
    runtime_.wait();
  }

  void Step() {
    for (size_t i = 0; i < N_ITER; i++) {
      k_substep_reset_grid_.launch();
      k_substep_p2g_.launch();
      k_substep_update_grid_v_.launch();
      k_substep_g2p_.launch();
    }
    runtime_.wait();
  }

 private:
  ti::Runtime runtime_;
  ti::AotModule module_;

  ti::NdArray<float> x_;
  ti::NdArray<float> v_;
  ti::NdArray<float> J_;
  ti::NdArray<float> C_;
  ti::NdArray<float> grid_v_;
  ti::NdArray<float> grid_m_;
  ti::NdArray<float> pos_;

  ti::Kernel k_init_particles_;
  ti::Kernel k_substep_reset_grid_;
  ti::Kernel k_substep_p2g_;
  ti::Kernel k_substep_update_grid_v_;
  ti::Kernel k_substep_g2p_;
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
