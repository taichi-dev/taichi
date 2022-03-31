#include "taichi/backends/vulkan/vulkan_program.h"
#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/aot_module_loader_impl.h"

#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include <GLFW/glfw3.h>

void aot_load() {
  // 1. Create a native window used by the Vulkan Runtime
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  GLFWwindow *window = glfwCreateWindow(512, 512, "Taichi show", NULL, NULL);
  if (window == NULL) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return;
  }

  // 2. Create an application config (not really used in this example)
  // required by other component to load and use the AOT modules
  taichi::ui::AppConfig app_config;
  app_config.name = "AOT Loader";
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.ti_arch = taichi::Arch::vulkan;
  app_config.is_packed_mode = true;
  app_config.width = 512;
  app_config.height = 512;

  // 3. Create the renderer required to initialize properly the Runtime
  std::unique_ptr<taichi::ui::vulkan::Renderer> renderer =
      std::make_unique<taichi::ui::vulkan::Renderer>();
  renderer->init(nullptr, (taichi::ui::TaichiWindow *)window, app_config);

  taichi::uint64 *result_buffer{nullptr};
  std::unique_ptr<taichi::lang::MemoryPool> memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(taichi::Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

  // 4. Create the Runtime
  taichi::lang::vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = &(renderer->app_context().device());
  std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // 5. Load the AOT module using the previously created Runtime
  taichi::lang::vulkan::AotModuleParams aotParams{".", vulkan_runtime.get()};
  std::unique_ptr<taichi::lang::aot::Module> module =
      taichi::lang::aot::Module::load(taichi::Arch::vulkan, aotParams);
  auto rootSize = module->get_root_size();
  vulkan_runtime->add_root_buffer(rootSize);

  taichi::lang::aot::Kernel *init_kernel = module->get_kernel("init");
  if (!init_kernel) {
    std::cerr << "Failed to load 'init' kernel" << std::endl;
    glfwTerminate();
    return;
  }

  // 6. Create a NdArray allocation that could be used by the AOT module kernel
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.size = 512 * sizeof(float);
  taichi::lang::DeviceAllocation dev_alloc =
      vulkan_runtime->get_ti_device()->allocate_memory(std::move(alloc_params));

  // 7. Execute the kernel loaded from the AOT module :) and setting the device
  // allocation as a parameter that could be used by this kernel
  taichi::lang::RuntimeContext host_ctx;
  host_ctx.set_arg(0, &dev_alloc);
  host_ctx.set_device_allocation(0, true);
  host_ctx.extra_args[0][0] = 1;
  host_ctx.extra_args[0][1] = 5;
  host_ctx.extra_args[0][2] = 1;

  init_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();
}
