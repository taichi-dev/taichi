#include "taichi_vulkan_impl.h"

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
    const taichi::lang::vulkan::VulkanDevice::Params &params)
    : vk_device_{} {
  vk_device_.init_vulkan_structs(params);
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
  params.api_version = VK_API_VERSION_1_2;
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
TiDevice ti_import_vulkan_device(const TiVulkanDeviceInteropInfo *interopInfo) {
  taichi::lang::vulkan::VulkanDevice::Params params{};
  params.instance = interopInfo->instance;
  params.physical_device = interopInfo->physicalDevice;
  params.device = interopInfo->device;
  params.compute_queue = interopInfo->computeQueue;
  params.compute_queue_family_index = interopInfo->computeQueueFamilyIndex;
  params.graphics_queue = interopInfo->graphicsQueue;
  params.graphics_queue_family_index = interopInfo->graphicsQueueFamilyIndex;
  return static_cast<Device *>(new VulkanDeviceImported(params));
}
void ti_export_vulkan_device(TiDevice device,
                          TiVulkanDeviceInteropInfo *interopInfo) {
  Device *device2 = (Device *)device;
  TI_ASSERT(device2->arch == taichi::Arch::vulkan);
  taichi::lang::vulkan::VulkanDevice &vk_device =
      static_cast<VulkanDevice *>(device2)->get_vk();
  interopInfo->instance = vk_device.vk_instance();
  interopInfo->physicalDevice = vk_device.vk_physical_device();
  interopInfo->device = vk_device.vk_device();
  interopInfo->computeQueue = vk_device.compute_queue();
  interopInfo->computeQueueFamilyIndex = vk_device.compute_queue_family_index();
  interopInfo->graphicsQueue = vk_device.graphics_queue();
  interopInfo->graphicsQueueFamilyIndex =
      vk_device.graphics_queue_family_index();
}

TiAotModule ti_load_vulkan_aot_module(TiContext context, const char *module_path) {
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
TiDeviceMemory ti_import_vulkan_deviceAllocation(
    TiDevice device,
    const TiVulkanDeviceAllocationInteropInfo *interopInfo) {
  Device *device2 = (Device *)device;
  TI_ASSERT(device2->arch == taichi::Arch::vulkan);

  taichi::lang::vulkan::VulkanDevice &vk_device =
      static_cast<VulkanDevice *>(device2)->get_vk();

  vkapi::IVkBuffer buffer =
      vkapi::create_buffer(vk_device.vk_device(), interopInfo->buffer);
  return (TiDeviceMemory)vk_device.import_vkbuffer(buffer).alloc_id;
}
void ti_export_vulkan_device_memory(
    TiDevice device,
    TiDeviceMemory deviceMemory,
    TiVulkanDeviceAllocationInteropInfo *interopInfo) {
  VulkanDevice *device2 = ((Device *)device)->as_vk();
  taichi::lang::DeviceAllocationId devalloc_id = deviceMemory;
  taichi::lang::DeviceAllocation devalloc{&device2->get(), devalloc_id};
  vkapi::IVkBuffer buffer = device2->get_vk().get_vkbuffer(devalloc);
  interopInfo->buffer = buffer.get()->buffer;
}

#endif  // TI_WITH_VULKAN
