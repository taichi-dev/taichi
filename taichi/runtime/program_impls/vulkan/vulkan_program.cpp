#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/aot/graph_data.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/util/offline_cache.h"

#include "taichi/rhi/window_system.h"

using namespace taichi::lang::vulkan;

namespace taichi::lang {

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

  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }
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

FunctionType register_params_to_executable(
    gfx::GfxRuntime::RegisterParams &&params,
    gfx::GfxRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(std::move(params));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace

VulkanProgramImpl::VulkanProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
}

FunctionType VulkanProgramImpl::compile(const CompileConfig &compile_config,
                                        Kernel *kernel) {
  return register_params_to_executable(
      get_cache_manager()->load_or_compile(compile_config, kernel),
      vulkan_runtime_.get());
}

void VulkanProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

// Android is meant to be embedded in other application only so the creation of
// the device and other states is left to the caller/host.
// The following code is only used when Taichi is running on its own.
#ifndef ANDROID
  GLFWwindow *glfw_window = nullptr;

  if (window_system::glfw_context_acquire()) {
    // glfw init success
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfw_window = glfwCreateWindow(1, 1, "Dummy Window", nullptr, nullptr);

    if (glfwVulkanSupported() != GLFW_TRUE) {
      TI_WARN("GLFW reports no Vulkan support");
    }
  }
#endif

  VulkanDeviceCreator::Params evd_params;
  if (config->vk_api_version.empty()) {
    // Don't assign the API version by default. Otherwise we have to provide all
    // the extensions to be enabled. `VulkanDeviceCreator` would automatically
    // select a usable version for us.
    evd_params.api_version = std::nullopt;
  } else {
    size_t idot1 = config->vk_api_version.find('.');
    size_t idot2 = config->vk_api_version.find('.', idot1 + 1);
    int32_t major = std::atoll(config->vk_api_version.c_str());
    int32_t minor = std::atoll(config->vk_api_version.c_str() + idot1 + 1);
    int32_t patch = std::atoll(config->vk_api_version.c_str() + idot2 + 1);
    evd_params.api_version = VK_MAKE_API_VERSION(0, major, minor, patch);
  }

  if (config->debug) {
    TI_WARN("Enabling vulkan validation layer in debug mode");
    evd_params.enable_validation_layer = true;
  }
#if !defined(ANDROID)
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

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = embedded_device_->device();
  vulkan_runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ =
      std::make_unique<gfx::SNodeTreeManager>(vulkan_runtime_.get());
}

void VulkanProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  if (vulkan_runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void VulkanProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                               uint64 *result_buffer) {
  snode_tree_mgr_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> VulkanProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (vulkan_runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(), Arch::vulkan, *config, caps);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, Arch::vulkan, *config, caps);
  }
}

DeviceAllocation VulkanProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  return get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false});
}

DeviceAllocation VulkanProgramImpl::allocate_texture(
    const ImageParams &params) {
  return vulkan_runtime_->create_image(params);
}

void VulkanProgramImpl::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  vulkan_runtime_->enqueue_compute_op_lambda(op, image_refs);
}

void VulkanProgramImpl::dump_cache_data_to_disk() {
  const auto &mgr = get_cache_manager();
  mgr->clean_offline_cache(offline_cache::string_to_clean_cache_policy(
                               config->offline_cache_cleaning_policy),
                           config->offline_cache_max_size_of_files,
                           config->offline_cache_cleaning_factor);
  mgr->dump_with_merging();
}

const std::unique_ptr<gfx::CacheManager>
    &VulkanProgramImpl::get_cache_manager() {
  if (!cache_manager_) {
    TI_ASSERT(vulkan_runtime_ && snode_tree_mgr_ && embedded_device_);
    using Mgr = gfx::CacheManager;
    Mgr::Params params;
    params.arch = config->arch;
    params.mode = config->offline_cache ? Mgr::MemAndDiskCache : Mgr::MemCache;
    params.cache_path = config->offline_cache_file_path;
    params.runtime = vulkan_runtime_.get();
    params.compile_config = config;
    params.caps = embedded_device_->device()->get_caps();
    params.compiled_structs = &snode_tree_mgr_->get_compiled_structs();
    cache_manager_ = std::make_unique<gfx::CacheManager>(std::move(params));
  }
  return cache_manager_;
}

VulkanProgramImpl::~VulkanProgramImpl() {
  cache_manager_.reset();
  vulkan_runtime_.reset();
  embedded_device_.reset();
}

}  // namespace taichi::lang
