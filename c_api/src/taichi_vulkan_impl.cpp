#include "taichi_core_impl.h"
#include "taichi_vulkan_impl.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "vulkan/vulkan.h"
#ifdef ANDROID
#define VK_KHR_android_surface 1
#include "vulkan/vulkan_android.h"
#endif

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
  vk_device.set_cap(taichi::lang::DeviceCapability::vk_api_version,
                    api_version);

  vk_device.set_cap(taichi::lang::DeviceCapability::spirv_version, 0x10000);
  if (api_version >= VK_API_VERSION_1_3) {
    vk_device.set_cap(taichi::lang::DeviceCapability::spirv_version, 0x10500);
  } else if (api_version >= VK_API_VERSION_1_2) {
    vk_device.set_cap(taichi::lang::DeviceCapability::spirv_version, 0x10500);
  } else if (api_version >= VK_API_VERSION_1_1) {
    vk_device.set_cap(taichi::lang::DeviceCapability::spirv_version, 0x10300);
  }

  if (api_version > VK_API_VERSION_1_0) {
    vk_device.set_cap(
        taichi::lang::DeviceCapability::spirv_has_physical_storage_buffer,
        true);
  }

  vk_device.init_vulkan_structs(
      const_cast<taichi::lang::vulkan::VulkanDevice::Params &>(params));
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
  params.api_version = std::nullopt;
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

TiTexture VulkanRuntime::allocate_texture(
    const taichi::lang::ImageParams &params) {
  taichi::lang::DeviceAllocation devalloc = get_vk().create_image(params);
  return devalloc2devtex(*this, devalloc);
}
void VulkanRuntime::free_texture(TiTexture texture) {
  get_vk().destroy_image(devtex2devalloc(*this, texture));
}

TiAotModule VulkanRuntime::load_aot_module(const char *module_path) {
  taichi::lang::gfx::AotModuleParams params{};
  params.module_path = module_path;
  params.runtime = &get_gfx_runtime();
  std::unique_ptr<taichi::lang::aot::Module> aot_module =
      taichi::lang::aot::Module::load(arch, params);
  size_t root_size = aot_module->get_root_size();
  params.runtime->add_root_buffer(root_size);
  return (TiAotModule)(new AotModule(*this, std::move(aot_module)));
}
void VulkanRuntime::buffer_copy(const taichi::lang::DevicePtr &dst,
                                const taichi::lang::DevicePtr &src,
                                size_t size) {
  get_gfx_runtime().buffer_copy(dst, src, size);
}
void VulkanRuntime::copy_image(const taichi::lang::DeviceAllocation &dst,
                               const taichi::lang::DeviceAllocation &src,
                               const taichi::lang::ImageCopyParams &params) {
  get_gfx_runtime().copy_image(dst, src, params);
}
void VulkanRuntime::transition_image(
    const taichi::lang::DeviceAllocation &image,
    taichi::lang::ImageLayout layout) {
  get_gfx_runtime().transition_image(image, layout);
}
void VulkanRuntime::submit() {
  get_gfx_runtime().flush();
}
void VulkanRuntime::signal_event(taichi::lang::DeviceEvent *event) {
  get_gfx_runtime().signal_event(event);
}
void VulkanRuntime::reset_event(taichi::lang::DeviceEvent *event) {
  get_gfx_runtime().reset_event(event);
}
void VulkanRuntime::wait_event(taichi::lang::DeviceEvent *event) {
  get_gfx_runtime().wait_event(event);
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
  if (api_version < VK_API_VERSION_1_0) {
    TI_WARN("ignored attempt to create vulkan runtime of version <1.0");
    return TI_NULL_HANDLE;
  }
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
  return (TiRuntime) static_cast<Runtime *>(new VulkanRuntimeOwned(params));
}
TiRuntime ti_import_vulkan_runtime(
    const TiVulkanRuntimeInteropInfo *interop_info) {
  if (interop_info->api_version < VK_API_VERSION_1_0) {
    TI_WARN("ignored attempt to import vulkan runtime of version <1.0");
    return TI_NULL_HANDLE;
  }
  if (interop_info->physical_device == nullptr) {
    TI_WARN(
        "ignored attempt to import vulkan runtime with vulkan physical device "
        "of null handle");
    return TI_NULL_HANDLE;
  }
  if (interop_info->device == nullptr) {
    TI_WARN(
        "ignored attempt to import vulkan runtime with vulkan device of null "
        "handle");
    return TI_NULL_HANDLE;
  }
  taichi::lang::vulkan::VulkanDevice::Params params{};
  params.instance = interop_info->instance;
  params.physical_device = interop_info->physical_device;
  params.device = interop_info->device;
  params.compute_queue = interop_info->compute_queue;
  params.compute_queue_family_index = interop_info->compute_queue_family_index;
  params.graphics_queue = interop_info->graphics_queue;
  params.graphics_queue_family_index =
      interop_info->graphics_queue_family_index;
  return (TiRuntime) static_cast<Runtime *>(
      new VulkanRuntimeImported(interop_info->api_version, params));
}
void ti_export_vulkan_runtime(TiRuntime runtime,
                              TiVulkanRuntimeInteropInfo *interop_info) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to export vulkan runtime of null handle");
    return;
  }
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

TiMemory ti_import_vulkan_memory(
    TiRuntime runtime,
    const TiVulkanMemoryInteropInfo *interop_info) {
  if (runtime == nullptr) {
    TI_WARN(
        "ignored attempt to import vulkan memory to runtime of null handle");
    return TI_NULL_HANDLE;
  }
  Runtime *runtime2 = (Runtime *)runtime;
  if (runtime2->arch != taichi::Arch::vulkan) {
    TI_WARN("ignored attempt to import vulkan memory to non-vulkan runtime");
    return TI_NULL_HANDLE;
  }
  taichi::lang::vulkan::VulkanDevice &vk_runtime =
      static_cast<VulkanRuntime *>(runtime2)->get_vk();

  vkapi::IVkBuffer buffer =
      vkapi::create_buffer(vk_runtime.vk_device(), interop_info->buffer,
                           interop_info->size, interop_info->usage);
  taichi::lang::DeviceAllocation devalloc = vk_runtime.import_vkbuffer(buffer);
  return devalloc2devmem(*runtime2, devalloc);
}
void ti_export_vulkan_memory(TiRuntime runtime,
                             TiMemory devmem,
                             TiVulkanMemoryInteropInfo *interop_info) {
  if (runtime == nullptr) {
    TI_WARN(
        "ignored attempt to export vulkan memory from runtime of null handle");
    return;
  }
  if (devmem == nullptr) {
    TI_WARN("ignored attempt to export vulkan memory of null handle");
    return;
  }
  VulkanRuntime *runtime2 = ((Runtime *)runtime)->as_vk();
  taichi::lang::DeviceAllocation devalloc = devmem2devalloc(*runtime2, devmem);
  vkapi::IVkBuffer buffer = runtime2->get_vk().get_vkbuffer(devalloc);
  interop_info->buffer = buffer.get()->buffer;
  interop_info->size = buffer.get()->size;
  interop_info->usage = buffer.get()->usage;
}
TiTexture ti_import_vulkan_texture(
    TiRuntime runtime,
    const TiVulkanTextureInteropInfo *interop_info,
    VkImageViewType view_type,
    VkImageLayout layout) {
  if (runtime == nullptr) {
    TI_WARN(
        "ignored attempt to import vulkan texture from runtime of null handle");
    return TI_NULL_HANDLE;
  }
  Runtime *runtime2 = ((Runtime *)runtime)->as_vk();
  if (runtime2->arch != taichi::Arch::vulkan) {
    TI_WARN("ignored attempt to import vulkan texture to non-vulkan runtime");
    return TI_NULL_HANDLE;
  }
  taichi::lang::vulkan::VulkanDevice &vk_runtime =
      static_cast<VulkanRuntime *>(runtime2)->get_vk();

  bool is_depth = interop_info->format == VK_FORMAT_D16_UNORM ||
                  interop_info->format == VK_FORMAT_D16_UNORM_S8_UINT ||
                  interop_info->format == VK_FORMAT_D24_UNORM_S8_UINT ||
                  interop_info->format == VK_FORMAT_D32_SFLOAT ||
                  interop_info->format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                  interop_info->format == VK_FORMAT_X8_D24_UNORM_PACK32;

  vkapi::IVkImage image =
      vkapi::create_image(vk_runtime.vk_device(), interop_info->image,
                          interop_info->format, interop_info->image_type,
                          interop_info->extent, interop_info->mip_level_count,
                          interop_info->array_layer_count, interop_info->usage);

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = nullptr;
  view_info.viewType = view_type;
  view_info.format = interop_info->format;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask =
      is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = interop_info->mip_level_count;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = interop_info->array_layer_count;

  vkapi::IVkImageView image_view =
      vkapi::create_image_view(vk_runtime.vk_device(), image, &view_info);

  taichi::lang::DeviceAllocation image2 =
      vk_runtime.import_vk_image(image, image_view, layout);
  return devalloc2devtex(*runtime2, image2);
}

void ti_export_vulkan_texture(TiRuntime runtime,
                              TiTexture texture,
                              TiVulkanTextureInteropInfo *interop_info) {
  if (runtime == nullptr) {
    TI_WARN(
        "ignored attempt to export vulkan texture from runtime of null handle");
    return;
  }
  if (texture == nullptr) {
    TI_WARN("ignored attempt to export vulkan texture of null handle");
    return;
  }
  VulkanRuntime *runtime2 = ((Runtime *)runtime)->as_vk();
  taichi::lang::DeviceAllocation devalloc = devtex2devalloc(*runtime2, texture);
  vkapi::IVkImage image =
      std::get<0>(runtime2->get_vk().get_vk_image(devalloc));
  interop_info->image = image->image;
  interop_info->image_type = image->type;
  interop_info->extent.width = image->width;
  interop_info->extent.height = image->height;
  interop_info->extent.depth = image->depth;
  interop_info->mip_level_count = image->mip_levels;
  interop_info->array_layer_count = image->array_layers;
  interop_info->sample_count = VK_SAMPLE_COUNT_1_BIT;
  interop_info->tiling = VK_IMAGE_TILING_OPTIMAL;
  interop_info->usage = image->usage;
}

TiEvent ti_import_vulkan_event(TiRuntime runtime,
                               const TiVulkanEventInteropInfo *interop_info) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to import vulkan event to runtime of null handle");
    return TI_NULL_HANDLE;
  }
  Runtime *runtime2 = (Runtime *)runtime;
  if (runtime2->arch != taichi::Arch::vulkan) {
    TI_WARN("ignored attempt to import vulkan memory to non-vulkan runtime");
    return TI_NULL_HANDLE;
  }

  vkapi::IVkEvent event = std::make_unique<vkapi::DeviceObjVkEvent>();
  event->device = runtime2->as_vk()->get_vk().vk_device();
  event->event = interop_info->event;
  event->external = true;

  std::unique_ptr<taichi::lang::DeviceEvent> event2(
      new taichi::lang::vulkan::VulkanDeviceEvent(std::move(event)));

  return (TiEvent) new Event(*runtime2, std::move(event2));
}
void ti_export_vulkan_event(TiRuntime runtime,
                            TiEvent event,
                            TiVulkanEventInteropInfo *interop_info) {
  if (runtime == nullptr) {
    TI_WARN(
        "ignored attempt to export vulkan memory from runtime of null handle");
    return;
  }
  if (event == nullptr) {
    TI_WARN("ignored attempt to export vulkan memory of null handle");
    return;
  }
  auto event2 =
      (taichi::lang::vulkan::VulkanDeviceEvent *)(&((Event *)event)->get());
  interop_info->event = event2->vkapi_ref->event;
}

#endif  // TI_WITH_VULKAN
