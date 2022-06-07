#include "taichi_vulkan_impl.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "vulkan/vulkan.h"

#ifdef TI_WITH_VULKAN

VulkanRuntime::VulkanRuntime() : Runtime(taichi::Arch::vulkan) {
}
taichi::lang::vulkan::VulkanDevice &VulkanRuntime::get_vk() {
  return static_cast<taichi::lang::vulkan::VulkanDevice &>(get());
}

VulkanRuntimeImported::Workaround::Workaround(
    uint32_t api_version,
    const taichi::lang::vulkan::VulkanDevice::Params &params)
    : vk_device{} {
  // FIXME: This part is copied from `vulkan_runtime_creator.cpp` which should
  // be refactorized I guess.
  if (!taichi::lang::vulkan::VulkanLoader::instance().init()) {
    throw std::runtime_error("Error loading vulkan");
  }
  taichi::lang::vulkan::VulkanLoader::instance().load_instance(params.instance);
  taichi::lang::vulkan::VulkanLoader::instance().load_device(params.device);
  vk_device.init_vulkan_structs(
      const_cast<taichi::lang::vulkan::VulkanDevice::Params &>(params));
  vk_device.set_cap(taichi::lang::DeviceCapability::vk_api_version,
                    api_version);
}
VulkanRuntimeImported::VulkanRuntimeImported(
    uint32_t api_version,
    const taichi::lang::vulkan::VulkanDevice::Params &params)
    : inner_(api_version, params),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{
          host_result_buffer_.data(), &inner_.vk_device}) {
}
taichi::lang::Device &VulkanRuntimeImported::get() {
  return static_cast<taichi::lang::Device &>(inner_.vk_device);
}

taichi::lang::vulkan::VulkanDeviceCreator::Params
make_vulkan_runtime_creator_params() {
#ifdef ANDROID
  const std::vector<std::string> extensions = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  };
#else
  std::vector<std::string> extensions = {
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
  };

  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.emplace_back(glfw_extensions[i]);
  }
#endif  // ANDROID

  // FIXME: (penguinliong) Vulkan runtime should be created outside.
  taichi::lang::vulkan::VulkanDeviceCreator::Params params{};
  params.api_version = VK_API_VERSION_1_3;
  params.additional_instance_extensions = extensions;
  params.additional_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  return params;
}
taichi::lang::gfx::GfxRuntime &VulkanRuntimeImported::get_gfx_runtime() {
  return gfx_runtime_;
}

VulkanRuntimeOwned::VulkanRuntimeOwned()
    : VulkanRuntimeOwned(make_vulkan_runtime_creator_params()) {
}
VulkanRuntimeOwned::VulkanRuntimeOwned(
    const taichi::lang::vulkan::VulkanDeviceCreator::Params &params)
    : vk_device_creator_(params),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{
          host_result_buffer_.data(), vk_device_creator_.device()}) {
}
taichi::lang::Device &VulkanRuntimeOwned::get() {
  return *static_cast<taichi::lang::Device *>(vk_device_creator_.device());
}
taichi::lang::gfx::GfxRuntime &VulkanRuntimeOwned::get_gfx_runtime() {
  return gfx_runtime_;
}

TiAotModule VulkanRuntime::load_aot_module(const char *module_path) {
  taichi::lang::gfx::AotModuleParams params{};
  params.module_path = module_path;
  params.runtime = &get_gfx_runtime();
  std::unique_ptr<taichi::lang::aot::Module> aot_module =
      taichi::lang::aot::Module::load(arch, params);
  size_t root_size = aot_module->get_root_size();
  params.runtime->add_root_buffer(root_size);
  return new AotModule(*this, std::move(aot_module));
}
void VulkanRuntime::submit() {
  get_gfx_runtime().flush();
}
void VulkanRuntime::wait() {
  // (penguinliong) It's currently waiting for the entire runtime to stop.
  // Should be simply waiting for its fence to finish.
  get_gfx_runtime().synchronize();
}

// -----------------------------------------------------------------------------

TiRuntime ti_create_vulkan_runtime_ext(uint32_t api_version,
                                       const char **instance_extensions,
                                       uint32_t instance_extensions_count,
                                       const char **device_extensions,
                                       uint32_t device_extensions_count) {
  taichi::lang::vulkan::VulkanDeviceCreator::Params params;
  params.api_version = api_version;
  params.is_for_ui = false;
  params.additional_instance_extensions.reserve(instance_extensions_count);
  for (uint32_t i = 0; i < instance_extensions_count; ++i) {
    params.additional_instance_extensions.push_back(instance_extensions[i]);
  }
  params.additional_device_extensions.reserve(device_extensions_count);
  for (uint32_t i = 0; i < device_extensions_count; ++i) {
    params.additional_device_extensions.push_back(device_extensions[i]);
  }
  params.surface_creator = nullptr;
  return static_cast<Runtime *>(new VulkanRuntimeOwned(params));
}
TiRuntime ti_import_vulkan_runtime(
    const TiVulkanRuntimeInteropInfo *interop_info) {
  taichi::lang::vulkan::VulkanDevice::Params params{};
  params.instance = interop_info->instance;
  params.physical_device = interop_info->physical_device;
  params.device = interop_info->device;
  params.compute_queue = interop_info->compute_queue;
  params.compute_queue_family_index = interop_info->compute_queue_family_index;
  params.graphics_queue = interop_info->graphics_queue;
  params.graphics_queue_family_index =
      interop_info->graphics_queue_family_index;
  return static_cast<Runtime *>(
      new VulkanRuntimeImported(interop_info->api_version, params));
}
void ti_export_vulkan_runtime(TiRuntime runtime,
                              TiVulkanRuntimeInteropInfo *interop_info) {
  Runtime *runtime2 = (Runtime *)runtime;
  TI_ASSERT(runtime2->arch == taichi::Arch::vulkan);
  taichi::lang::vulkan::VulkanDevice &vk_device =
      static_cast<VulkanRuntime *>(runtime2)->get_vk();
  interop_info->api_version =
      vk_device.get_cap(taichi::lang::DeviceCapability::vk_api_version);
  interop_info->instance = vk_device.vk_instance();
  interop_info->physical_device = vk_device.vk_physical_device();
  interop_info->device = vk_device.vk_device();
  interop_info->compute_queue = vk_device.compute_queue();
  interop_info->compute_queue_family_index =
      vk_device.compute_queue_family_index();
  interop_info->graphics_queue = vk_device.graphics_queue();
  interop_info->graphics_queue_family_index =
      vk_device.graphics_queue_family_index();
}

TiMemory ti_import_vulkan_runtime_allocation(
    TiRuntime runtime,
    const TiVulkanMemoryInteropInfo *interop_info) {
  Runtime *runtime2 = (Runtime *)runtime;
  TI_ASSERT(runtime2->arch == taichi::Arch::vulkan);

  taichi::lang::vulkan::VulkanDevice &vk_runtime =
      static_cast<VulkanRuntime *>(runtime2)->get_vk();

  vkapi::IVkBuffer buffer =
      vkapi::create_buffer(vk_runtime.vk_device(), interop_info->buffer,
                           interop_info->size, interop_info->usage);
  return (TiMemory)vk_runtime.import_vkbuffer(buffer).alloc_id;
}
void ti_export_vulkan_memory(TiRuntime runtime,
                             TiMemory devmem,
                             TiVulkanMemoryInteropInfo *interop_info) {
  VulkanRuntime *runtime2 = ((Runtime *)runtime)->as_vk();
  taichi::lang::DeviceAllocationId devalloc_id = devmem;
  taichi::lang::DeviceAllocation devalloc{&runtime2->get(), devalloc_id};
  vkapi::IVkBuffer buffer = runtime2->get_vk().get_vkbuffer(devalloc);
  interop_info->buffer = buffer.get()->buffer;
  interop_info->size = buffer.get()->size;
  interop_info->usage = buffer.get()->usage;
}

#endif  // TI_WITH_VULKAN
