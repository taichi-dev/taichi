#include "taichi/backends/vulkan/vulkan_program.h"

#include "GLFW/glfw3.h"

using namespace taichi::lang::vulkan;

namespace taichi {
namespace lang {

namespace {
std::vector<std::string> get_required_instance_extensions() {
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<std::string> extensions;

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }

  // EmbeddedVulkanDevice will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#if TI_WITH_CUDA
  // so that we can do cuda-vk interop
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif  // TI_WITH_CUDA
  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if TI_WITH_CUDA
        // so that we can do cuda-vk interop
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef _WIN64
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
#endif  // TI_WITH_CUDA
  };

  return extensions;
}
}  // namespace

FunctionType VulkanProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  vulkan::lower(kernel);
  return vulkan::compile_to_executable(kernel, vulkan_runtime_.get());
}

void VulkanProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  GLFWwindow *glfw_window = nullptr;
  if (glfwInit()) {
    // glfw init success
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
    glfw_window = glfwCreateWindow(1, 1, "Dummy Window", nullptr, nullptr);
  }

  EmbeddedVulkanDevice::Params evd_params;
  evd_params.api_version = VulkanEnvSettings::kApiVersion();
  if (glfw_window) {
    // then we should be able to create a device with graphics abilities
    evd_params.additional_instance_extensions =
        get_required_instance_extensions();
    evd_params.additional_device_extensions = get_required_device_extensions();
    evd_params.is_for_ui = true;
    evd_params.surface_creator = [&](VkInstance instance) -> VkSurfaceKHR {
      VkSurfaceKHR surface = VK_NULL_HANDLE;
      if (glfwCreateWindowSurface(instance, glfw_window, nullptr, &surface) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
      }
      return surface;
    };
  }

  embedded_device_ = std::make_unique<EmbeddedVulkanDevice>(evd_params);

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = embedded_device_->device();
  vulkan_runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void VulkanProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    uint64 *result_buffer) {
  vulkan_runtime_->materialize_snode_tree(tree);
}

VulkanProgramImpl::~VulkanProgramImpl() {
  vulkan_runtime_.reset();
  embedded_device_.reset();
}

}  // namespace lang
}  // namespace taichi
