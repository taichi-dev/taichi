#include <chrono>
#include <iostream>
#include <signal.h>
#include <inttypes.h>
#include <unistd.h>

#include "gtest/gtest.h"

#include "mpm88_test.hpp"

#include "c_api_test_utils.h"
#include "taichi/taichi_vulkan.h"

#if defined(TI_WITH_LLVM) && defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)

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
    const std::vector<int> vec2_shape = {2};
    const std::vector<int> vec3_shape = {3};
    const std::vector<int> mat2_shape = {2, 2};

    x_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                             {kNrParticles}, vec2_shape,
                             /*host_read=*/false, /*host_write=*/false);

    v_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                             {kNrParticles}, vec2_shape);

    pos_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                               {kNrParticles}, vec3_shape, false, false);

    C_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                             {kNrParticles}, mat2_shape);

    J_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                             {kNrParticles}, {});

    grid_v_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                  {kNGrid, kNGrid}, vec2_shape);
    grid_m_ = NdarrayAndMem::Make(runtime_, TiDataType::TI_DATA_TYPE_F32,
                                  {kNGrid, kNGrid}, {});

    k_init_particles_ = ti_get_aot_module_kernel(module_, "init_particles");
    k_substep_g2p_ = ti_get_aot_module_kernel(module_, "substep_g2p");
    k_substep_reset_grid_ =
        ti_get_aot_module_kernel(module_, "substep_reset_grid");
    k_substep_p2g_ = ti_get_aot_module_kernel(module_, "substep_p2g");
    k_substep_update_grid_v_ =
        ti_get_aot_module_kernel(module_, "substep_update_grid_v");

    k_init_particles_args_[0] = x_->argument();
    k_init_particles_args_[1] = v_->argument();
    k_init_particles_args_[2] = J_->argument();

    k_substep_reset_grid_args_[0] = grid_v_->argument();
    k_substep_reset_grid_args_[1] = grid_m_->argument();

    k_substep_p2g_args_[0] = x_->argument();
    k_substep_p2g_args_[1] = v_->argument();
    k_substep_p2g_args_[2] = C_->argument();
    k_substep_p2g_args_[3] = J_->argument();
    k_substep_p2g_args_[4] = grid_v_->argument();
    k_substep_p2g_args_[5] = grid_m_->argument();

    k_substep_update_grid_v_args_[0] = grid_v_->argument();
    k_substep_update_grid_v_args_[1] = grid_m_->argument();

    k_substep_g2p_args_[0] = x_->argument();
    k_substep_g2p_args_[1] = v_->argument();
    k_substep_g2p_args_[2] = C_->argument();
    k_substep_g2p_args_[3] = J_->argument();
    k_substep_g2p_args_[4] = grid_v_->argument();
    k_substep_g2p_args_[5] = pos_->argument();

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
  class NdarrayAndMem {
   public:
    NdarrayAndMem() = default;
    ~NdarrayAndMem() {
    }

    const TiArgument &argument() const {
      return arr_arg_;
    }

    static std::unique_ptr<NdarrayAndMem> Make(
        TiRuntime runtime,
        TiDataType dtype,
        const std::vector<int> &arr_shape,
        const std::vector<int> &element_shape = {},
        bool host_read = false,
        bool host_write = false) {
      // TODO: Cannot use data_type_size() until
      // https://github.com/taichi-dev/taichi/pull/5220.
      // uint64_t_t alloc_size = taichi::lang::data_type_size(dtype);
      uint64_t alloc_size = 4;
      assert(dtype == TiDataType::TI_DATA_TYPE_F32 ||
             dtype == TiDataType::TI_DATA_TYPE_I32 ||
             dtype == TiDataType::TI_DATA_TYPE_U32);
      alloc_size = 4;

      for (int s : arr_shape) {
        alloc_size *= s;
      }
      for (int s : element_shape) {
        alloc_size *= s;
      }

      auto res = std::make_unique<NdarrayAndMem>();
      res->runtime_ = runtime;

      TiMemoryAllocateInfo alloc_info;
      alloc_info.size = alloc_size;
      alloc_info.host_write = false;
      alloc_info.host_read = false;
      alloc_info.export_sharing = false;
      alloc_info.usage = TiMemoryUsageFlagBits::TI_MEMORY_USAGE_STORAGE_BIT;

      res->memory_ = ti_allocate_memory(res->runtime_, &alloc_info);

      TiNdShape shape;
      shape.dim_count = static_cast<uint32_t>(arr_shape.size());
      for (size_t i = 0; i < arr_shape.size(); i++) {
        shape.dims[i] = arr_shape[i];
      }

      TiNdShape e_shape;
      e_shape.dim_count = static_cast<uint32_t>(element_shape.size());
      for (size_t i = 0; i < element_shape.size(); i++) {
        e_shape.dims[i] = element_shape[i];
      }

      TiNdArray arg_array = {.memory = res->memory_,
                             .shape = std::move(shape),
                             .elem_shape = std::move(e_shape),
                             .elem_type = dtype};

      TiArgumentValue arg_value = {.ndarray = std::move(arg_array)};

      res->arr_arg_ = {.type = TiArgumentType::TI_ARGUMENT_TYPE_NDARRAY,
                       .value = std::move(arg_value)};

      return res;
    }

   private:
    TiRuntime runtime_;
    TiMemory memory_;
    TiArgument arr_arg_;
  };

  void InitTaichiRuntime(TiArch arch) {
    runtime_ = ti_create_runtime(arch);
  }

  TiRuntime runtime_;
  TiAotModule module_{nullptr};
  TiVulkanRuntimeInteropInfo interop_info;

  std::unique_ptr<NdarrayAndMem> x_{nullptr};
  std::unique_ptr<NdarrayAndMem> v_{nullptr};
  std::unique_ptr<NdarrayAndMem> J_{nullptr};
  std::unique_ptr<NdarrayAndMem> C_{nullptr};
  std::unique_ptr<NdarrayAndMem> grid_v_{nullptr};
  std::unique_ptr<NdarrayAndMem> grid_m_{nullptr};
  std::unique_ptr<NdarrayAndMem> pos_{nullptr};

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

MPM88Demo::MPM88Demo(const std::string &aot_path, TiArch arch) {
  // Create Taichi Device for computation
  impl_ = std::make_unique<MPM88DemoImpl>(aot_path, arch);
}

void MPM88Demo::Step() {
  impl_->Step();
}

MPM88Demo::~MPM88Demo() {
  impl_.reset();
}

}  // namespace demo

TEST(CapiMpm88Test, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto mpm88_demo = std::make_unique<demo::MPM88Demo>(
        aot_mod_ss.str().c_str(), TiArch::TI_ARCH_CUDA);
    mpm88_demo->Step();
  }
}

TEST(CapiMpm88Test, Vulkan) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto mpm88_demo = std::make_unique<demo::MPM88Demo>(
        aot_mod_ss.str().c_str(), TiArch::TI_ARCH_VULKAN);
    mpm88_demo->Step();
  }
}
#endif
