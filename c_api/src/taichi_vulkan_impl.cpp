#include "taichi_vulkan_impl.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "vulkan/vulkan.h"

#ifdef TI_WITH_VULKAN

VulkanDevice::VulkanDevice() : Device(taichi::Arch::vulkan) {
}
taichi::lang::vulkan::VulkanDevice &VulkanDevice::get_vk() {
  return static_cast<taichi::lang::vulkan::VulkanDevice &>(get());
}

Context *VulkanDevice::create_context() {
  return new VulkanContext(*this);
}

VulkanDeviceImported::VulkanDeviceImported(
    uint32_t api_version,
    const taichi::lang::vulkan::VulkanDevice::Params &params)
    : vk_device_{} {
  // FIXME: This part is copied from `vulkan_device_creator.cpp` which should
  // be refactorized I guess.
  if (!taichi::lang::vulkan::VulkanLoader::instance().init()) {
    throw std::runtime_error("Error loading vulkan");
  }
  taichi::lang::vulkan::VulkanLoader::instance().load_instance(params.instance);
  taichi::lang::vulkan::VulkanLoader::instance().load_device(params.device);
  vk_device_.init_vulkan_structs(
      const_cast<taichi::lang::vulkan::VulkanDevice::Params &>(params));
  vk_device_.set_cap(taichi::lang::DeviceCapability::vk_api_version,
                     api_version);
}
taichi::lang::Device &VulkanDeviceImported::get() {
  return static_cast<taichi::lang::Device &>(vk_device_);
}

taichi::lang::vulkan::VulkanDeviceCreator::Params
make_vulkan_device_creator_params() {
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

  // FIXME: (penguinliong) Vulkan device should be created outside.
  taichi::lang::vulkan::VulkanDeviceCreator::Params params{};
  params.api_version = VK_API_VERSION_1_3;
  params.additional_instance_extensions = extensions;
  params.additional_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  return params;
}

VulkanDeviceOwned::VulkanDeviceOwned()
    : VulkanDeviceOwned(make_vulkan_device_creator_params()) {
}
VulkanDeviceOwned::VulkanDeviceOwned(
    const taichi::lang::vulkan::VulkanDeviceCreator::Params &params)
    : vk_device_creator_(params) {
}
taichi::lang::Device &VulkanDeviceOwned::get() {
  return *static_cast<taichi::lang::Device *>(vk_device_creator_.device());
}

VulkanContext::VulkanContext(VulkanDevice &device)
    : Context(device),
      host_result_buffer_(),
      vk_runtime_(taichi::lang::vulkan::VkRuntime::Params{
          host_result_buffer_.data(), &device.get()}) {
}
VulkanContext::~VulkanContext() {
}

void VulkanContext::submit() {
  vk_runtime_.flush();
}
void VulkanContext::wait() {
  // (penguinliong) It's currently waiting for the entire device to stop. Should
  // be simply waiting for its fence to finish.
  vk_runtime_.synchronize();
}

taichi::lang::vulkan::VkRuntime &VulkanContext::get_vk() {
  return vk_runtime_;
}

// -----------------------------------------------------------------------------

TiDevice ti_create_vulkan_device_ext(uint32_t api_version,
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
  return static_cast<Device *>(new VulkanDeviceOwned(params));
}
TiDevice ti_import_vulkan_device(
    const TiVulkanDeviceInteropInfo *interop_info) {
  taichi::lang::vulkan::VulkanDevice::Params params{};
  params.instance = interop_info->instance;
  params.physical_device = interop_info->physical_device;
  params.device = interop_info->device;
  params.compute_queue = interop_info->compute_queue;
  params.compute_queue_family_index = interop_info->compute_queue_family_index;
  params.graphics_queue = interop_info->graphics_queue;
  params.graphics_queue_family_index =
      interop_info->graphics_queue_family_index;
  return static_cast<Device *>(
      new VulkanDeviceImported(interop_info->api_version, params));
}
void ti_export_vulkan_device(TiDevice device,
                             TiVulkanDeviceInteropInfo *interop_info) {
  Device *device2 = (Device *)device;
  TI_ASSERT(device2->arch == taichi::Arch::vulkan);
  taichi::lang::vulkan::VulkanDevice &vk_device =
      static_cast<VulkanDevice *>(device2)->get_vk();
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

TiAotModule ti_load_vulkan_aot_module(TiContext context,
                                      const char *module_path) {
  VulkanContext *context2 = ((Context *)context)->as_vk();
  taichi::lang::vulkan::VkRuntime &vk_runtime = context2->get_vk();
  taichi::lang::vulkan::AotModuleParams params{};
  params.module_path = module_path;
  params.runtime = &vk_runtime;
  std::unique_ptr<taichi::lang::aot::Module> aot_module =
      taichi::lang::aot::Module::load(context2->device().arch, params);
  size_t root_size = aot_module->get_root_size();
  vk_runtime.add_root_buffer(root_size);
  return new AotModule(*context2, std::move(aot_module));
}
TiDeviceMemory ti_import_vulkan_device_allocation(
    TiDevice device,
    const TiVulkanDeviceMemoryInteropInfo *interop_info) {
  Device *device2 = (Device *)device;
  TI_ASSERT(device2->arch == taichi::Arch::vulkan);

  taichi::lang::vulkan::VulkanDevice &vk_device =
      static_cast<VulkanDevice *>(device2)->get_vk();

  vkapi::IVkBuffer buffer =
      vkapi::create_buffer(vk_device.vk_device(), interop_info->buffer,
                           interop_info->size, interop_info->usage);
  return (TiDeviceMemory)vk_device.import_vkbuffer(buffer).alloc_id;
}
void ti_export_vulkan_device_memory(
    TiDevice device,
    TiDeviceMemory devmem,
    TiVulkanDeviceMemoryInteropInfo *interop_info) {
  VulkanDevice *device2 = ((Device *)device)->as_vk();
  taichi::lang::DeviceAllocationId devalloc_id = devmem;
  taichi::lang::DeviceAllocation devalloc{&device2->get(), devalloc_id};
  vkapi::IVkBuffer buffer = device2->get_vk().get_vkbuffer(devalloc);
  interop_info->buffer = buffer.get()->buffer;
  interop_info->size = buffer.get()->size;
  interop_info->usage = buffer.get()->usage;
}

#endif  // TI_WITH_VULKAN
