#ifdef TI_WITH_VULKAN
#include "taichi_vulkan_impl.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/common/utils.h"

#ifdef ANDROID
#define VK_KHR_android_surface 1
#include "vulkan/vulkan_android.h"
#else
#include "GLFW/glfw3.h"
#endif  // ANDROID

VulkanRuntime::VulkanRuntime() : GfxRuntime(taichi::Arch::vulkan) {
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
  if (!taichi::lang::vulkan::VulkanLoader::instance().init(
          params.get_proc_addr)) {
    throw std::runtime_error("Error loading vulkan");
  }
  taichi::lang::vulkan::VulkanLoader::instance().load_instance(params.instance);
  taichi::lang::vulkan::VulkanLoader::instance().load_device(params.device);
  vk_device.vk_caps().vk_api_version = api_version;
  // FIXME: (penguinliong) Workaround missing vulkan caps from import.
  vk_device.vk_caps().external_memory = true;

  taichi::lang::DeviceCapabilityConfig caps{};

  if (api_version >= VK_API_VERSION_1_2) {
    caps.set(taichi::lang::DeviceCapability::spirv_version, 0x10500);
  } else if (api_version >= VK_API_VERSION_1_1) {
    caps.set(taichi::lang::DeviceCapability::spirv_version, 0x10300);
  } else {
    caps.set(taichi::lang::DeviceCapability::spirv_version, 0x10000);
  }

  // (penguinliong) Will bring it back after devcap.
  /*
  if (api_version > VK_API_VERSION_1_0) {
    caps.set(taichi::lang::DeviceCapability::spirv_has_physical_storage_buffer,
             true);
  }
  */

  vk_device.set_caps(std::move(caps));
  vk_device.init_vulkan_structs(
      const_cast<taichi::lang::vulkan::VulkanDevice::Params &>(params));
}
VulkanRuntimeImported::VulkanRuntimeImported(
    uint32_t api_version,
    const taichi::lang::vulkan::VulkanDevice::Params &params)
    : inner_(api_version, params),
      gfx_runtime_(taichi::lang::gfx::GfxRuntime::Params{&inner_.vk_device}) {
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
      gfx_runtime_(
          taichi::lang::gfx::GfxRuntime::Params{vk_device_creator_.device()}) {
}
taichi::lang::Device &VulkanRuntimeOwned::get() {
  return *static_cast<taichi::lang::Device *>(vk_device_creator_.device());
}
taichi::lang::gfx::GfxRuntime &VulkanRuntimeOwned::get_gfx_runtime() {
  return gfx_runtime_;
}

TiImage VulkanRuntime::allocate_image(const taichi::lang::ImageParams &params) {
  taichi::lang::DeviceAllocation devalloc =
      get_gfx_runtime().create_image(params);
  return devalloc2devimg(*this, devalloc);
}
void VulkanRuntime::free_image(TiImage image) {
  taichi::lang::DeviceAllocation devimg = devimg2devalloc(*this, image);
  get_vk().destroy_image(devimg);
  get_gfx_runtime().untrack_image(devimg);
}

// -----------------------------------------------------------------------------

TiRuntime ti_create_vulkan_runtime_ext(uint32_t api_version,
                                       uint32_t instance_extension_count,
                                       const char **instance_extensions,
                                       uint32_t device_extension_count,
                                       const char **device_extensions) {
  TiRuntime out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  if (api_version < VK_API_VERSION_1_0) {
    ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "api_version<1.0");
    return TI_NULL_HANDLE;
  }
  if (instance_extension_count > 0) {
    TI_CAPI_ARGUMENT_NULL_RV(instance_extensions);
  }
  if (device_extension_count > 0) {
    TI_CAPI_ARGUMENT_NULL_RV(device_extensions);
  }

  taichi::lang::vulkan::VulkanDeviceCreator::Params params;
  params.api_version = api_version;
  params.is_for_ui = false;
  params.additional_instance_extensions.reserve(instance_extension_count);
  for (uint32_t i = 0; i < instance_extension_count; ++i) {
    TI_CAPI_ARGUMENT_NULL_RV(instance_extensions[i]);
    params.additional_instance_extensions.push_back(instance_extensions[i]);
  }
  params.additional_device_extensions.reserve(device_extension_count);
  for (uint32_t i = 0; i < device_extension_count; ++i) {
    TI_CAPI_ARGUMENT_NULL_RV(device_extensions[i]);
    params.additional_device_extensions.push_back(device_extensions[i]);
  }
  params.surface_creator = nullptr;
  if (is_ci()) {
    params.enable_validation_layer = true;
  }
  out = (TiRuntime) static_cast<Runtime *>(new VulkanRuntimeOwned(params));
  TI_CAPI_TRY_CATCH_END();
  return out;
}
TiRuntime ti_import_vulkan_runtime(
    const TiVulkanRuntimeInteropInfo *interop_info) {
  TiRuntime out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(interop_info);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info->instance);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info->physical_device);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info->device);

  taichi::lang::vulkan::VulkanDevice::Params params{};
  params.get_proc_addr = interop_info->get_instance_proc_addr;
  params.instance = interop_info->instance;
  params.physical_device = interop_info->physical_device;
  params.device = interop_info->device;
  params.compute_queue = interop_info->compute_queue;
  params.compute_queue_family_index = interop_info->compute_queue_family_index;
  params.graphics_queue = interop_info->graphics_queue;
  params.graphics_queue_family_index =
      interop_info->graphics_queue_family_index;
  out = (TiRuntime) static_cast<Runtime *>(
      new VulkanRuntimeImported(interop_info->api_version, params));
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_export_vulkan_runtime(TiRuntime runtime,
                              TiVulkanRuntimeInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(interop_info);

  Runtime *runtime2 = (Runtime *)runtime;
  taichi::lang::vulkan::VulkanDevice &vk_device =
      static_cast<VulkanRuntime *>(runtime2)->get_vk();
  interop_info->get_instance_proc_addr = vkGetInstanceProcAddr;
  interop_info->api_version = vk_device.vk_caps().vk_api_version;
  interop_info->instance = vk_device.vk_instance();
  interop_info->physical_device = vk_device.vk_physical_device();
  interop_info->device = vk_device.vk_device();
  interop_info->compute_queue = vk_device.compute_queue();
  interop_info->compute_queue_family_index =
      vk_device.compute_queue_family_index();
  interop_info->graphics_queue = vk_device.graphics_queue();
  interop_info->graphics_queue_family_index =
      vk_device.graphics_queue_family_index();
  TI_CAPI_TRY_CATCH_END();
}

TiMemory ti_import_vulkan_memory(
    TiRuntime runtime,
    const TiVulkanMemoryInteropInfo *interop_info) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info->buffer);
  TI_CAPI_INVALID_INTEROP_ARCH_RV(((Runtime *)runtime)->arch, vulkan);

  Runtime *runtime2 = (Runtime *)runtime;
  taichi::lang::vulkan::VulkanDevice &vk_runtime =
      static_cast<VulkanRuntime *>(runtime2)->get_vk();

  vkapi::IVkBuffer buffer = vkapi::create_buffer(
      vk_runtime.vk_device(), interop_info->buffer, interop_info->usage);
  taichi::lang::DeviceAllocation devalloc = vk_runtime.import_vkbuffer(
      buffer, interop_info->size, interop_info->memory, interop_info->offset);
  out = devalloc2devmem(*runtime2, devalloc);
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_export_vulkan_memory(TiRuntime runtime,
                             TiMemory memory,
                             TiVulkanMemoryInteropInfo *interop_info) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(memory);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  TI_CAPI_INVALID_INTEROP_ARCH(((Runtime *)runtime)->arch, vulkan);

  VulkanRuntime *runtime2 = ((Runtime *)runtime)->as_vk();
  taichi::lang::DeviceAllocation devalloc = devmem2devalloc(*runtime2, memory);
  vkapi::IVkBuffer buffer = runtime2->get_vk().get_vkbuffer(devalloc);

  auto [vk_mem, offset, size] =
      runtime2->get_vk().get_vkmemory_offset_size(devalloc);

  interop_info->buffer = buffer.get()->buffer;
  interop_info->size = size;
  interop_info->usage = buffer.get()->usage;
  interop_info->memory = vk_mem;
  interop_info->offset = (uint64_t)offset;
}
TiImage ti_import_vulkan_image(TiRuntime runtime,
                               const TiVulkanImageInteropInfo *interop_info,
                               VkImageViewType view_type,
                               VkImageLayout layout) {
  TiImage out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info);
  TI_CAPI_ARGUMENT_NULL_RV(interop_info->image);
  TI_CAPI_INVALID_INTEROP_ARCH_RV(((Runtime *)runtime)->arch, vulkan);

  Runtime *runtime2 = ((Runtime *)runtime)->as_vk();
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

  taichi::lang::ImageLayout layout2 = (taichi::lang::ImageLayout)layout;
  static_cast<VulkanRuntime *>(runtime2)->track_image(image2, layout2);

  out = devalloc2devimg(*runtime2, image2);
  TI_CAPI_TRY_CATCH_END();
  return out;
}

void ti_export_vulkan_image(TiRuntime runtime,
                            TiImage image,
                            TiVulkanImageInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(image);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  TI_CAPI_INVALID_INTEROP_ARCH(((Runtime *)runtime)->arch, vulkan);

  VulkanRuntime *runtime2 = ((Runtime *)runtime)->as_vk();

  taichi::lang::DeviceAllocation devalloc = devimg2devalloc(*runtime2, image);
  vkapi::IVkImage image2 =
      std::get<0>(runtime2->get_vk().get_vk_image(devalloc));
  interop_info->image = image2->image;
  interop_info->image_type = image2->type;
  interop_info->format = image2->format;
  interop_info->extent.width = image2->width;
  interop_info->extent.height = image2->height;
  interop_info->extent.depth = image2->depth;
  interop_info->mip_level_count = image2->mip_levels;
  interop_info->array_layer_count = image2->array_layers;
  interop_info->sample_count = VK_SAMPLE_COUNT_1_BIT;
  interop_info->tiling = VK_IMAGE_TILING_OPTIMAL;
  interop_info->usage = image2->usage;
  TI_CAPI_TRY_CATCH_END();
}

#endif  // TI_WITH_VULKAN
