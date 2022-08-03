#include <chrono>
#include <iostream>
#include <signal.h>
#include <inttypes.h>
#include <unistd.h>

#include "gtest/gtest.h"

#include "mpm88_test.hpp"

#include "c_api_test_utils.h"
#include "taichi_core_impl.h"
#include "taichi/taichi_core.h"
#include "taichi/taichi_vulkan.h"

#if defined(TI_WITH_LLVM) && defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)

namespace demo {
namespace {
constexpr int kNrParticles = 8192 * 2;
constexpr int kNGrid = 128;
constexpr size_t N_ITER = 50;

static taichi::Arch get_taichi_arch(const std::string &arch_name_) {
  if (arch_name_ == "cuda") {
    return taichi::Arch::cuda;
  }

  if (arch_name_ == "x64") {
    return taichi::Arch::x64;
  }

  if (arch_name_ == "vulkan") {
    return taichi::Arch::vulkan;
  }

  TI_ERROR("Unkown arch_name");
  return taichi::Arch::x64;
}

static TiArch get_c_api_arch(const std::string &arch_name_) {
  if (arch_name_ == "cuda") {
    return TiArch::TI_ARCH_CUDA;
  }

  if (arch_name_ == "x64") {
    return TiArch::TI_ARCH_X64;
  }

  if (arch_name_ == "vulkan") {
    return TiArch::TI_ARCH_VULKAN;
  }

  TI_ERROR("Unkown arch_name");
  return TiArch::TI_ARCH_X64;
}

static taichi::ui::FieldSource get_field_source(const std::string &arch_name_) {
  if (arch_name_ == "cuda") {
    return taichi::ui::FieldSource::TaichiCuda;
  }

  if (arch_name_ == "x64") {
    return taichi::ui::FieldSource::TaichiX64;
  }

  if (arch_name_ == "vulkan") {
    return taichi::ui::FieldSource::TaichiVulkan;
  }

  TI_ERROR("Unkown arch_name");
  return taichi::ui::FieldSource::TaichiX64;
}
}  // namespace

class MPM88DemoImpl {
 public:
  MPM88DemoImpl(const std::string &aot_path,
                TiArch arch,
                taichi::lang::vulkan::VulkanDevice *vk_device) {
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
                               {kNrParticles}, vec3_shape, false, false,
                               vk_device);

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

  taichi::lang::DeviceAllocation &pos() {
    return pos_->devalloc_;
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

    taichi::lang::DeviceAllocation devalloc() {
      Runtime *real_runtime = (Runtime *)runtime_;
      return devmem2devalloc(*real_runtime, memory_);
    }

    static std::unique_ptr<NdarrayAndMem> Make(
        TiRuntime runtime,
        TiDataType dtype,
        const std::vector<int> &arr_shape,
        const std::vector<int> &element_shape = {},
        bool host_read = false,
        bool host_write = false,
        taichi::lang::vulkan::VulkanDevice *vk_device_ = nullptr) {
      // TODO: Cannot use data_type_size() until
      // https://github.com/taichi-dev/taichi/pull/5220.
      // uint64_t_t alloc_size = taichi::lang::data_type_size(dtype);
      uint64_t alloc_size = 1;
      TI_ASSERT(dtype == TiDataType::TI_DATA_TYPE_F32 ||
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

      if (!vk_device_) {
        TiMemoryAllocateInfo alloc_info;
        alloc_info.size = alloc_size;
        alloc_info.host_write = false;
        alloc_info.host_read = false;
        alloc_info.export_sharing = false;
        alloc_info.usage = TiMemoryUsageFlagBits::TI_MEMORY_USAGE_STORAGE_BIT;

        res->memory_ = ti_allocate_memory(res->runtime_, &alloc_info);
        res->devalloc_ = res->devalloc();

      } else {
        taichi::lang::Device::AllocParams alloc_params;
        alloc_params.host_read = false;
        alloc_params.host_write = false;
        alloc_params.size = alloc_size;
        alloc_params.usage = taichi::lang::AllocUsage::Storage;

        res->devalloc_ = vk_device_->allocate_memory(alloc_params);

        res->interop_info.buffer =
            vk_device_->get_vkbuffer(res->devalloc_).get()->buffer;
        res->interop_info.size =
            vk_device_->get_vkbuffer(res->devalloc_).get()->size;
        res->interop_info.usage =
            vk_device_->get_vkbuffer(res->devalloc_).get()->usage;

        res->memory_ =
            ti_import_vulkan_memory(res->runtime_, &res->interop_info);
      }

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

    TiVulkanMemoryInteropInfo interop_info;
    taichi::lang::DeviceAllocation devalloc_;

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

MPM88Demo::MPM88Demo(const std::string &aot_path,
                     const std::string &arch_name) {
  // Init gl window
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(512, 512, "Taichi show", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
  }

  // Create a GGUI configuration
  taichi::ui::AppConfig app_config;
  app_config.name = "MPM88";
  app_config.width = 512;
  app_config.height = 512;
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.package_path = "../python/taichi";  // make it flexible later
  app_config.ti_arch = get_taichi_arch(arch_name);
  app_config.is_packed_mode = true;

  // Create GUI & renderer
  renderer = std::make_unique<taichi::ui::vulkan::Renderer>();
  renderer->init(nullptr, window, app_config);

  renderer->set_background_color({0.6, 0.6, 0.6});

  gui_ = std::make_shared<taichi::ui::vulkan::Gui>(
      &renderer->app_context(), &renderer->swap_chain(), window);

  taichi::lang::vulkan::VulkanDevice *device = nullptr;
  if (arch_name == "vulkan") {
    device = &(renderer->app_context().device());
  }

  // Create Taichi Device for computation
  impl_ = std::make_unique<MPM88DemoImpl>(aot_path, get_c_api_arch(arch_name),
                                          device);

  // Describe information to render the circle with Vulkan
  f_info.valid = true;
  f_info.field_type = taichi::ui::FieldType::Scalar;
  f_info.matrix_rows = 1;
  f_info.matrix_cols = 1;
  f_info.shape = {kNrParticles};
  f_info.field_source = get_field_source(arch_name);
  f_info.dtype = taichi::lang::PrimitiveType::f32;
  f_info.snode = nullptr;
  f_info.dev_alloc = impl_->pos();

  circles.renderable_info.has_per_vertex_color = false;
  circles.renderable_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;
  circles.renderable_info.vbo = f_info;
  circles.color = {0.8, 0.4, 0.1};
  circles.radius = 0.005f;  // 0.0015f looks unclear on desktop
}

void MPM88Demo::Step() {
  // while (!glfwWindowShouldClose(window)) {
  for (size_t i = 0; i < 10; i++) {
    impl_->Step();

    // Render elements
    renderer->circles(circles);
    renderer->draw_frame(gui_.get());
    renderer->swap_chain().surface().present_image();
    renderer->prepare_for_next_frame();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  //}
}

MPM88Demo::~MPM88Demo() {
  impl_.reset();
  gui_.reset();
  // renderer owns the device so it must be destructed last.
  renderer.reset();
}

}  // namespace demo

TEST(CapiMpm88Test, Cuda) {
  if (capi::utils::is_cuda_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto mpm88_demo =
        std::make_unique<demo::MPM88Demo>(aot_mod_ss.str().c_str(), "cuda");
    mpm88_demo->Step();
  }
}

TEST(CapiMpm88Test, Vulkan) {
  if (capi::utils::is_vulkan_available()) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;

    auto mpm88_demo =
        std::make_unique<demo::MPM88Demo>(aot_mod_ss.str().c_str(), "vulkan");
    mpm88_demo->Step();
  }
}
#endif
