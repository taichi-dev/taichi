#include "taichi/backends/vulkan/vulkan_program.h"
#include "taichi/backends/vulkan/aot_module_builder_impl.h"

#if !defined(ANDROID) && !defined(TI_EMSCRIPTENED)
#include "GLFW/glfw3.h"
#endif

using namespace taichi::lang::vulkan;

namespace taichi {
namespace lang {

namespace {
std::vector<std::string> get_required_instance_extensions() {
#ifdef ANDROID
  std::vector<std::string> extensions;

  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
#else
  std::vector<std::string> extensions;

#ifndef TI_EMSCRIPTENED
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }
#endif
  // VulkanDeviceCreator will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#if TI_WITH_CUDA
  // so that we can do cuda-vk interop
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif  // TI_WITH_CUDA
  return extensions;
#endif
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

VulkanProgramImpl::VulkanProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
}

FunctionType compile_to_executable(Kernel *kernel, VkRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(std::move(run_codegen(
      kernel, runtime->get_ti_device(), runtime->get_compiled_structs())));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

FunctionType VulkanProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  spirv::lower(kernel);
  return compile_to_executable(kernel, vulkan_runtime_.get());
}

void VulkanProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

#ifndef TI_EMSCRIPTENED
// Android is meant to be embedded in other application only so the creation of
// the device and other states is left to the caller/host.
// The following code is only used when Taichi is running on its own.
#ifndef ANDROID
  GLFWwindow *glfw_window = nullptr;
#ifdef __APPLE__
  glfwInitVulkanLoader(vkGetInstanceProcAddr);
#endif

  if (glfwInit()) {
    // glfw init success
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
    glfw_window = glfwCreateWindow(1, 1, "Dummy Window", nullptr, nullptr);

    if (glfwVulkanSupported() != GLFW_TRUE) {
      TI_WARN("GLFW reports no Vulkan support");
    }
  }
#endif
#endif

  VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = VulkanEnvSettings::kApiVersion();
#if !defined(ANDROID) && !defined(TI_EMSCRIPTENED)
  if (glfw_window) {
    // then we should be able to create a device with graphics abilities
    evd_params.additional_instance_extensions =
        get_required_instance_extensions();
    evd_params.additional_device_extensions = get_required_device_extensions();
    evd_params.is_for_ui = true;
    evd_params.surface_creator = [&](VkInstance instance) -> VkSurfaceKHR {
      VkSurfaceKHR surface = VK_NULL_HANDLE;
      TI_TRACE("before glfwCreateWindowSurface {} {}", (void *)glfw_window,
               (void *)instance);
      uint status = VK_SUCCESS;
      if ((status = glfwCreateWindowSurface(instance, glfw_window, nullptr,
                                            &surface)) != VK_SUCCESS) {
        TI_ERROR("Failed to create window surface! err: {}", status);
        throw std::runtime_error("failed to create window surface!");
      }
      return surface;
    };
  }
#endif

  embedded_device_ = std::make_unique<VulkanDeviceCreator>(evd_params);

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = embedded_device_->device();
  vulkan_runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void VulkanProgramImpl::compile_snode_tree_types(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees) {
  if (vulkan_runtime_) {
    vulkan_runtime_->materialize_snode_tree(tree);
  } else {
    CompiledSNodeStructs compiled_structs =
        vulkan::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void VulkanProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    uint64 *result_buffer) {
  vulkan_runtime_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> VulkanProgramImpl::make_aot_module_builder() {
  if (vulkan_runtime_) {
    return std::make_unique<AotModuleBuilderImpl>(
        vulkan_runtime_->get_compiled_structs());
  } else {
    return std::make_unique<AotModuleBuilderImpl>(aot_compiled_snode_structs_);
  }
}

DeviceAllocation VulkanProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  auto &ndarray =
      ref_ndarry_.emplace_back(get_compute_device()->allocate_memory_unique(
          {alloc_size, /*host_write=*/false, /*host_read=*/false,
           /*export_sharing=*/false}));
  return *ndarray;
}

VulkanProgramImpl::~VulkanProgramImpl() {
  ref_ndarry_.clear();
  vulkan_runtime_.reset();
  embedded_device_.reset();
}

}  // namespace lang
}  // namespace taichi
