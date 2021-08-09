#include "taichi/backends/vulkan/vulkan_api.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/backends/vulkan/loader.h"
#include "taichi/common/logging.h"

namespace taichi {
namespace lang {
namespace vulkan {

namespace {

constexpr bool kEnableValidationLayers = true;
const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

bool check_validation_layer_support() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  std::unordered_set<std::string> available_layer_names;
  for (const auto &layer_props : available_layers) {
    available_layer_names.insert(layer_props.layerName);
  }
  for (const char *name : kValidationLayers) {
    if (available_layer_names.count(std::string(name)) == 0) {
      return false;
    }
  }
  return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                  VkDebugUtilsMessageTypeFlagsEXT message_type,
                  const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
                  void *p_user_data) {
  if (message_severity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    TI_WARN("validation layer: {}", p_callback_data->pMessage);
  }
  return VK_FALSE;
}

void populate_debug_messenger_create_info(
    VkDebugUtilsMessengerCreateInfoEXT *create_info) {
  *create_info = {};
  create_info->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info->pfnUserCallback = vk_debug_callback;
  create_info->pUserData = nullptr;
}

VkResult create_debug_utils_messenger_ext(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT *p_create_info,
    const VkAllocationCallbacks *p_allocator,
    VkDebugUtilsMessengerEXT *p_debug_messenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, p_create_info, p_allocator, p_debug_messenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void destroy_debug_utils_messenger_ext(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debug_messenger,
    const VkAllocationCallbacks *p_allocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debug_messenger, p_allocator);
  }
}

std::vector<const char *> get_required_extensions() {
  std::vector<const char *> extensions;
  if constexpr (kEnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

VulkanQueueFamilyIndices find_queue_families(VkPhysicalDevice device,
                                             VkSurfaceKHR surface) {
  VulkanQueueFamilyIndices indices;

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           queue_families.data());
  // TODO: What the heck is this?
  constexpr VkQueueFlags kFlagMask =
      (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT));

  // first try and find a queue that has just the compute bit set
  // FIXME: Actually create two queues (async compute & graphics if supported)
  for (int i = 0; i < (int)queue_family_count; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if ((masked_flags & VK_QUEUE_COMPUTE_BIT) &&
        (masked_flags & VK_QUEUE_GRAPHICS_BIT)) {
      indices.compute_family = i;
    }
    if (masked_flags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }

    if (surface != VK_NULL_HANDLE) {
      VkBool32 present_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                           &present_support);

      if (present_support) {
        indices.present_family = i;
      }
    }

    if (indices.is_complete() && indices.is_complete_for_ui()) {
      return indices;
    }
  }

  // lastly get any queue that will work
  for (int i = 0; i < (int)queue_family_count; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if (masked_flags & VK_QUEUE_COMPUTE_BIT) {
      indices.compute_family = i;
    }
    if (indices.is_complete()) {
      return indices;
    }
  }
  return indices;
}

bool is_device_suitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
  auto indices = find_queue_families(device, surface);
  if (surface != VK_NULL_HANDLE) {
    return indices.is_complete_for_ui();
  } else {
    return indices.is_complete();
  }
}

VkShaderModule create_shader_module(VkDevice device,
                                    const SpirvCodeView &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size;
  create_info.pCode = code.data;

  VkShaderModule shader_module;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateShaderModule(device, &create_info, kNoVkAllocCallbacks,
                           &shader_module),
      "failed to create shader module");
  return shader_module;
}

}  // namespace

VulkanDevice::VulkanDevice(const Params &params) : rep_(params) {
}

EmbeddedVulkanDevice::EmbeddedVulkanDevice(const Params &params)
    : params_(params) {
  if (!VulkanLoader::instance().init()) {
    throw std::runtime_error("Error loading vulkan");
  }
  create_instance();
  setup_debug_messenger();
  if (params_.is_for_ui) {
    create_surface();
  }
  pick_physical_device();
  create_logical_device();
  create_command_pool();
  create_debug_swapchain();

  VulkanDevice::Params dparams;
  dparams.device = device_;
  dparams.compute_queue = compute_queue_;
  dparams.graphics_queue = graphics_queue_;
  dparams.present_queue = present_queue_;
  dparams.command_pool = command_pool_;
  owned_device_ = std::make_unique<VulkanDevice>(dparams);
#ifdef TI_VULKAN_DEBUG
  owned_device_->set_debug_struct(&debug_struct_);
#endif
}

EmbeddedVulkanDevice::~EmbeddedVulkanDevice() {
#ifdef TI_VULKAN_DEBUG
  if (capability_.has_presentation) {
    vkDestroySemaphore(device_, debug_struct_.image_available,
                       kNoVkAllocCallbacks);
    vkDestroySwapchainKHR(device_, debug_struct_.swapchain,
                          kNoVkAllocCallbacks);
    vkDestroySurfaceKHR(instance_, debug_struct_.surface, kNoVkAllocCallbacks);
    glfwDestroyWindow(debug_struct_.window);
    glfwTerminate();
  }
#endif

  if constexpr (kEnableValidationLayers) {
    destroy_debug_utils_messenger_ext(instance_, debug_messenger_,
                                      kNoVkAllocCallbacks);
  }
  vkDestroyCommandPool(device_, command_pool_, kNoVkAllocCallbacks);
  vkDestroyDevice(device_, kNoVkAllocCallbacks);
  vkDestroyInstance(instance_, kNoVkAllocCallbacks);
}

void EmbeddedVulkanDevice::create_instance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Taichi Vulkan Backend";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);

  if (params_.api_version.has_value()) {
    // The version client specified to use
    app_info.apiVersion = params_.api_version.value();
  } else {
    // The highest version designed to use
    app_info.apiVersion = VK_API_VERSION_1_2;
  }

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  if constexpr (kEnableValidationLayers) {
    TI_ASSERT_INFO(check_validation_layer_support(),
                   "validation layers requested but not available");
  }

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  if constexpr (kEnableValidationLayers) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();

    populate_debug_messenger_create_info(&debug_create_info);
    create_info.pNext = &debug_create_info;
  } else {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  auto extensions = get_required_extensions();
  for (auto ext : params_.additional_instance_extensions) {
    extensions.push_back(ext);
  }

#ifdef TI_VULKAN_DEBUG
  glfwInit();
  uint32_t count;
  const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&count);
  for (uint32_t i = 0; i < count; i++) {
    extensions.push_back(glfw_extensions[i]);
  }
#endif

  create_info.enabledExtensionCount = (uint32_t)extensions.size();
  create_info.ppEnabledExtensionNames = extensions.data();

  VkResult res =
      vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_);

  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkApplicationInfo.html
    // Vulkan 1.0 implementation will return this when api version is not 1.0
    // Vulkan 1.1+ implementation will work with maximum version set
    app_info.apiVersion = VK_API_VERSION_1_0;

    res = vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_);
  }

  if (res != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance");
  }
  VulkanLoader::instance().load_instance(instance_);
}

void EmbeddedVulkanDevice::setup_debug_messenger() {
  if constexpr (!kEnableValidationLayers) {
    return;
  }
  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  populate_debug_messenger_create_info(&create_info);

  BAIL_ON_VK_BAD_RESULT(
      create_debug_utils_messenger_ext(instance_, &create_info,
                                       kNoVkAllocCallbacks, &debug_messenger_),
      "failed to set up debug messenger");
}

void EmbeddedVulkanDevice::create_surface() {
  surface_ = params_.surface_creator(instance_);
}

void EmbeddedVulkanDevice::pick_physical_device() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  TI_ASSERT_INFO(device_count > 0, "failed to find GPUs with Vulkan support");

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
  physical_device_ = VK_NULL_HANDLE;
  for (const auto &device : devices) {
    if (is_device_suitable(device, surface_)) {
      physical_device_ = device;
      break;
    }
  }
  TI_ASSERT_INFO(physical_device_ != VK_NULL_HANDLE,
                 "failed to find a suitable GPU");

  queue_family_indices_ = find_queue_families(physical_device_, surface_);
}

void EmbeddedVulkanDevice::create_logical_device() {
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::unordered_set<uint32_t> unique_families;

  if (params_.is_for_ui) {
    unique_families = {queue_family_indices_.graphics_family.value(),
                       queue_family_indices_.present_family.value()};
  } else {
    unique_families = {queue_family_indices_.compute_family.value()};
  }

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_families) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queue_family;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queueCreateInfo);
  }

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.queueCreateInfoCount = queue_create_infos.size();

  // Get device properties
  VkPhysicalDeviceProperties physical_device_properties;
  vkGetPhysicalDeviceProperties(physical_device_, &physical_device_properties);
  capability_.api_version = physical_device_properties.apiVersion;
  capability_.spirv_version = 0x10000;

  if (capability_.api_version >= VK_API_VERSION_1_1) {
    capability_.spirv_version = 0x10300;
  }

  // Detect extensions
  std::vector<const char *> enabled_extensions;

  for (auto ext : params_.additional_device_extensions) {
    enabled_extensions.push_back(ext);
  }

  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr,
                                       &extension_count, nullptr);
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  vkEnumerateDeviceExtensionProperties(
      physical_device_, nullptr, &extension_count, extension_properties.data());

  bool has_surface = false, has_swapchain = false;

  for (auto &ext : extension_properties) {
    TI_TRACE("Vulkan device extension {} ({})", ext.extensionName,
             ext.specVersion);

    std::string name = std::string(ext.extensionName);

    if (name == "VK_KHR_portability_subset") {
      TI_WARN(
          "Potential non-conformant Vulkan implementation, enabling "
          "VK_KHR_portability_subset");
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SURFACE_EXTENSION_NAME) {
      has_surface = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
      has_swapchain = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) {
      capability_.has_atomic_float = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) {
      capability_.has_atomic_i64 = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SPIRV_1_4_EXTENSION_NAME) {
      if (capability_.spirv_version < 0x10400) {
        capability_.spirv_version = 0x10400;
        enabled_extensions.push_back(ext.extensionName);
      }
    } else if (name == VK_NV_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME) {
      capability_.has_nvidia_interop = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME) {
      capability_.has_spv_variable_ptr = true;
      enabled_extensions.push_back(ext.extensionName);
    }
  }

  if (has_surface && has_swapchain) {
    capability_.has_presentation = true;
  }

  TI_WARN_IF(
      !capability_.has_spv_variable_ptr,
      "Taichi may generate kernels that requires VK_KHR_VARIABLE_POINTERS, but "
      "this extension is not supported on the device");

  VkPhysicalDeviceFeatures device_features{};
  device_features.shaderInt16 = false;
  device_features.shaderInt64 = true;
  device_features.shaderFloat64 = true;
  capability_.has_int8 = false;
  capability_.has_int16 = false;
  capability_.has_int64 = true;
  capability_.has_float64 = true;
  if (params_.is_for_ui) {
    device_features.samplerAnisotropy = VK_TRUE;
    device_features.geometryShader = VK_TRUE;
    device_features.wideLines = VK_TRUE;
  }

  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount = enabled_extensions.size();
  create_info.ppEnabledExtensionNames = enabled_extensions.data();

  void **pNextEnd = (void **)&create_info.pNext;

  // TODO: Figure out whether to use this pNext chain or the Vulkan11 features
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VUID-VkDeviceCreateInfo-pNext-02829
  VkPhysicalDeviceVariablePointersFeatures variable_ptr_feature{};
  variable_ptr_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES;
  variable_ptr_feature.pNext = nullptr;

  if (capability_.has_spv_variable_ptr) {
    variable_ptr_feature.variablePointers = true;
    variable_ptr_feature.variablePointersStorageBuffer = true;
    *pNextEnd = &variable_ptr_feature;
    pNextEnd = &variable_ptr_feature.pNext;
  }

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shader_atomic_float_feature{};
  shader_atomic_float_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
  shader_atomic_float_feature.pNext = nullptr;

  if (capability_.has_atomic_float) {
    shader_atomic_float_feature.shaderBufferFloat32Atomics = true;
    shader_atomic_float_feature.shaderBufferFloat32AtomicAdd = true;
    shader_atomic_float_feature.shaderBufferFloat64Atomics = true;
    shader_atomic_float_feature.shaderBufferFloat64AtomicAdd = true;
    shader_atomic_float_feature.shaderSharedFloat32Atomics = true;
    shader_atomic_float_feature.shaderSharedFloat32AtomicAdd = true;
    shader_atomic_float_feature.shaderSharedFloat64Atomics = true;
    shader_atomic_float_feature.shaderSharedFloat64AtomicAdd = true;
    shader_atomic_float_feature.shaderImageFloat32Atomics = true;
    shader_atomic_float_feature.shaderImageFloat32AtomicAdd = true;
    shader_atomic_float_feature.sparseImageFloat32Atomics = true;
    shader_atomic_float_feature.sparseImageFloat32AtomicAdd = true;
    *pNextEnd = &shader_atomic_float_feature;
    pNextEnd = &shader_atomic_float_feature.pNext;
  }

  if constexpr (kEnableValidationLayers) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }
  BAIL_ON_VK_BAD_RESULT(vkCreateDevice(physical_device_, &create_info,
                                       kNoVkAllocCallbacks, &device_),
                        "failed to create logical device");
  VulkanLoader::instance().load_device(device_);

  if (params_.is_for_ui) {
    vkGetDeviceQueue(device_, queue_family_indices_.graphics_family.value(), 0,
                     &graphics_queue_);
    vkGetDeviceQueue(device_, queue_family_indices_.graphics_family.value(), 0,
                     &present_queue_);
  } else {
    vkGetDeviceQueue(device_, queue_family_indices_.compute_family.value(), 0,
                     &compute_queue_);
  }
}

void EmbeddedVulkanDevice::create_command_pool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = 0;
  if (params_.is_for_ui) {
    pool_info.queueFamilyIndex = queue_family_indices_.graphics_family.value();
  } else {
    pool_info.queueFamilyIndex = queue_family_indices_.compute_family.value();
  }
  BAIL_ON_VK_BAD_RESULT(
      vkCreateCommandPool(device_, &pool_info, kNoVkAllocCallbacks,
                          &command_pool_),
      "failed to create command pool");
}

void VulkanDevice::debug_frame_marker() const {
#ifdef TI_VULKAN_DEBUG
  if (debug_struct_) {
    uint32_t imageIndex;
    vkAcquireNextImageKHR(rep_.device, debug_struct_->swapchain, UINT64_MAX,
                          debug_struct_->image_available, VK_NULL_HANDLE,
                          &imageIndex);

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &debug_struct_->image_available;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &debug_struct_->swapchain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    vkQueuePresentKHR(rep_.compute_queue, &presentInfo);
  }
#endif
}

void EmbeddedVulkanDevice::create_debug_swapchain() {
#ifdef TI_VULKAN_DEBUG
  TI_TRACE("Creating debug swapchian");
  if (capability_.has_presentation) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    debug_struct_.window =
        glfwCreateWindow(640, 480, "Taichi Debug Swapchain", NULL, NULL);
    VkResult err = glfwCreateWindowSurface(instance_, debug_struct_.window,
                                           NULL, &debug_struct_.surface);
    if (err) {
      TI_ERROR("Failed to create debug window ({})", err);
      return;
    }

    auto choose_surface_format =
        [](const std::vector<VkSurfaceFormatKHR> &availableFormats) {
          for (const auto &availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
                availableFormat.colorSpace ==
                    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
              return availableFormat;
            }
          }
          return availableFormats[0];
        };

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device_, debug_struct_.surface, &capabilities);

    VkBool32 supported = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(
        physical_device_, queue_family_indices_.compute_family.value(),
        debug_struct_.surface, &supported);

    if (!supported) {
      TI_ERROR("Selected queue does not support presenting", err);
      return;
    }

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device_, debug_struct_.surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surface_formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_,
                                         debug_struct_.surface, &formatCount,
                                         surface_formats.data());

    VkSurfaceFormatKHR surface_format = choose_surface_format(surface_formats);

    int width, height;
    glfwGetFramebufferSize(debug_struct_.window, &width, &height);

    VkExtent2D extent = {uint32_t(width), uint32_t(height)};

    VkSwapchainCreateInfoKHR createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    createInfo.surface = debug_struct_.surface;
    createInfo.minImageCount = capabilities.minImageCount;
    createInfo.imageFormat = surface_format.format;
    createInfo.imageColorSpace = surface_format.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = nullptr;

    if (vkCreateSwapchainKHR(device_, &createInfo, kNoVkAllocCallbacks,
                             &debug_struct_.swapchain) != VK_SUCCESS) {
      TI_ERROR("Failed to create debug swapchain");
      return;
    }

    VkSemaphoreCreateInfo sema_create_info;
    sema_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sema_create_info.pNext = nullptr;
    sema_create_info.flags = 0;
    vkCreateSemaphore(device_, &sema_create_info, kNoVkAllocCallbacks,
                      &debug_struct_.image_available);
    TI_TRACE("Creating debug swapchian3");
  }
#endif
}

VulkanPipeline::VulkanPipeline(const Params &params)
    : device_(params.device->device()), name_(params.name) {
  create_descriptor_set_layout(params);
  create_compute_pipeline(params);
  create_descriptor_pool(params);
  create_descriptor_sets(params);
}

VulkanPipeline::~VulkanPipeline() {
  vkDestroyDescriptorPool(device_, descriptor_pool_, kNoVkAllocCallbacks);
  vkDestroyPipeline(device_, pipeline_, kNoVkAllocCallbacks);
  vkDestroyPipelineLayout(device_, pipeline_layout_, kNoVkAllocCallbacks);
  vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_,
                               kNoVkAllocCallbacks);
}

void VulkanPipeline::create_descriptor_set_layout(const Params &params) {
  const auto &buffer_binds = params.buffer_bindings;
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
  layout_bindings.reserve(buffer_binds.size());
  for (const auto &bb : buffer_binds) {
    VkDescriptorSetLayoutBinding layout_binding{};
    layout_binding.binding = bb.binding;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layout_binding.pImmutableSamplers = nullptr;
    layout_bindings.push_back(layout_binding);
  }

  VkDescriptorSetLayoutCreateInfo layout_create_info{};
  layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_create_info.bindingCount = layout_bindings.size();
  layout_create_info.pBindings = layout_bindings.data();

  BAIL_ON_VK_BAD_RESULT(
      vkCreateDescriptorSetLayout(device_, &layout_create_info,
                                  kNoVkAllocCallbacks, &descriptor_set_layout_),
      "failed to create descriptor set layout");
}

void VulkanPipeline::create_compute_pipeline(const Params &params) {
  VkShaderModule shader_module = create_shader_module(device_, params.code);

  VkPipelineShaderStageCreateInfo shader_stage_info{};
  shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage_info.module = shader_module;
#pragma message("Shader storage info: pName is hardcoded to \"main\"")
  shader_stage_info.pName = "main";

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;
  BAIL_ON_VK_BAD_RESULT(
      vkCreatePipelineLayout(device_, &pipeline_layout_info,
                             kNoVkAllocCallbacks, &pipeline_layout_),
      "failed to create pipeline layout");

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = shader_stage_info;
  pipeline_info.layout = pipeline_layout_;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateComputePipelines(device_, /*pipelineCache=*/VK_NULL_HANDLE,
                               /*createInfoCount=*/1, &pipeline_info,
                               kNoVkAllocCallbacks, &pipeline_),
      "failed to create pipeline");

  vkDestroyShaderModule(device_, shader_module, kNoVkAllocCallbacks);
}

void VulkanPipeline::create_descriptor_pool(const Params &params) {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  // This is the total number of descriptors we will allocate from this pool,
  // across all the descriptor sets.
  // https://stackoverflow.com/a/51716660/12003165
  pool_size.descriptorCount = params.buffer_bindings.size();

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateDescriptorPool(device_, &pool_info, kNoVkAllocCallbacks,
                             &descriptor_pool_),
      "failed to create descriptor pool");
}

void VulkanPipeline::create_descriptor_sets(const Params &params) {
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout_;

  BAIL_ON_VK_BAD_RESULT(
      vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_),
      "failed to allocate descriptor set");

  const auto &buffer_binds = params.buffer_bindings;
  std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
  descriptor_buffer_infos.reserve(buffer_binds.size());
  for (const auto &bb : buffer_binds) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = bb.buffer;
    // Note that this is the offset within the buffer itself, not the offset
    // of this buffer within its backing memory!
    buffer_info.offset = 0;
    // https://github.com/apache/tvm/blob/d288bbc5df3660355adbf97f2f84ecd232e269ff/src/runtime/vulkan/vulkan.cc#L1073
    buffer_info.range = VK_WHOLE_SIZE;
    descriptor_buffer_infos.push_back(buffer_info);
  }

  std::vector<VkWriteDescriptorSet> descriptor_writes;
  descriptor_writes.reserve(descriptor_buffer_infos.size());
  for (int i = 0; i < buffer_binds.size(); ++i) {
    const auto &bb = buffer_binds[i];

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = bb.binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptor_buffer_infos[i];
    write.pImageInfo = nullptr;
    write.pTexelBufferView = nullptr;
    descriptor_writes.push_back(write);
  }

  vkUpdateDescriptorSets(device_,
                         /*descriptorWriteCount=*/descriptor_writes.size(),
                         descriptor_writes.data(), /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);
}

VulkanCommandBuilder::VulkanCommandBuilder(const VulkanDevice *device) {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = device->command_pool();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  BAIL_ON_VK_BAD_RESULT(
      vkAllocateCommandBuffers(device->device(), &alloc_info, &command_buffer_),
      "failed to allocate command buffer");

  this->device_ = device->device();

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  // This flag allows us to submit the same command buffer to the queue
  // multiple times, while they are still pending.
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  begin_info.pInheritanceInfo = nullptr;
  BAIL_ON_VK_BAD_RESULT(vkBeginCommandBuffer(command_buffer_, &begin_info),
                        "failed to begin recording command buffer");
}

VulkanCommandBuilder::~VulkanCommandBuilder() {
  if (command_buffer_ != VK_NULL_HANDLE) {
    build();
  }
}

VkCommandBuffer VulkanCommandBuilder::build() {
  BAIL_ON_VK_BAD_RESULT(vkEndCommandBuffer(command_buffer_),
                        "failed to record command buffer");
  VkCommandBuffer res = command_buffer_;
  command_buffer_ = VK_NULL_HANDLE;
  return res;
}

void VulkanCommandBuilder::dispatch(const VulkanPipeline &pipeline,
                                    int group_count_x) {
  // Must call extension functions through a function pointer:
  PFN_vkCmdBeginDebugUtilsLabelEXT pfnCmdBeginDebugUtilsLabelEXT =
      (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(
          device_, "vkCmdBeginDebugUtilsLabelEXT");
  PFN_vkCmdEndDebugUtilsLabelEXT pfnCmdEndDebugUtilsLabelEXT =
      (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(
          device_, "vkCmdEndDebugUtilsLabelEXT");

  VkDebugUtilsLabelEXT marker;
  marker.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  marker.pNext = nullptr;
  marker.color[0] = 1.0f;
  marker.color[1] = 0.7f;
  marker.color[2] = 0.3f;
  marker.color[3] = 1.0f;
  marker.pLabelName = pipeline.name().data();
  pfnCmdBeginDebugUtilsLabelEXT(command_buffer_, &marker);

  vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.pipeline());
  vkCmdBindDescriptorSets(
      command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline.pipeline_layout(),
      /*firstSet=*/0, /*descriptorSetCount=*/1, &(pipeline.descriptor_set()),
      /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/nullptr);
  vkCmdDispatch(command_buffer_, group_count_x,
                /*groupCountY=*/1,
                /*groupCountZ=*/1);
  // Copied from TVM
  // https://github.com/apache/tvm/blob/b2a3c481ebbb7cfbd5335fb11cd516ae5f348406/src/runtime/vulkan/vulkan.cc#L1134-L1142
  VkMemoryBarrier barrier_info{};
  barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier_info.pNext = nullptr;
  barrier_info.srcAccessMask =
      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  barrier_info.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command_buffer_,
                       /*srcStageMask=*/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       /*srcStageMask=*/0, /*memoryBarrierCount=*/1,
                       &barrier_info, /*bufferMemoryBarrierCount=*/0,
                       /*pBufferMemoryBarriers=*/nullptr,
                       /*imageMemoryBarrierCount=*/0,
                       /*pImageMemoryBarriers=*/nullptr);

  pfnCmdEndDebugUtilsLabelEXT(command_buffer_);
}

void VulkanCommandBuilder::copy(VkBuffer src_buffer,
                                VkBuffer dst_buffer,
                                VkDeviceSize size,
                                VulkanCopyBufferDirection direction) {
  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;
  vkCmdCopyBuffer(command_buffer_, src_buffer, dst_buffer, /*regionCount=*/1,
                  &copy_region);
  if (direction == VulkanCopyBufferDirection::H2D) {
    VkMemoryBarrier barrier_info;
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;

    barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_info.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_,
                         /*srcStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT,
                         /*dstStageMask=*/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier_info, 0, nullptr, 0, nullptr);
  }
}

VkCommandBuffer record_copy_buffer_command(
    const VulkanDevice *device,
    VkBuffer src_buffer,
    VkBuffer dst_buffer,
    VkDeviceSize size,
    VulkanCopyBufferDirection direction) {
  VulkanCommandBuilder cb{device};
  cb.copy(src_buffer, dst_buffer, size, direction);
  return cb.build();
}

VulkanStream::VulkanStream(const VulkanDevice *device) : device_(device) {
}

void VulkanStream::launch(VkCommandBuffer command) {
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command;

  VkFence fence;
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
  vkCreateFence(device_->device(), &fence_info, kNoVkAllocCallbacks, &fence);
  in_flight_fences_.push_back(fence);

  BAIL_ON_VK_BAD_RESULT(
      vkQueueSubmit(device_->compute_queue(), /*submitCount=*/1, &submit_info,
                    /*fence=*/fence),
      "failed to submit command buffer");
}

void VulkanStream::synchronize() {
  // While vkQueueWaitIdle is strongly discouraged, this is probably the most
  // viable way for synchronization in Taichi. Unlike graphics pipeline, there
  // is no clear boundary (i.e. frame) for us to use a VkFence. TVM accumulates
  // all the commands into a single buffer, then submits it all at once upon
  // synchronization. Not sure how efficient that model is.

  // vkQueueWaitIdle(device_->compute_queue());

  if (in_flight_fences_.size()) {
    vkWaitForFences(device_->device(), in_flight_fences_.size(),
                    in_flight_fences_.data(), true, 0xFFFFFFFF);
    for (auto &fence : in_flight_fences_) {
      vkDestroyFence(device_->device(), fence, kNoVkAllocCallbacks);
    }
    in_flight_fences_.clear();
  }

  device_->debug_frame_marker();
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
