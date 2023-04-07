#include "taichi/rhi/vulkan/vulkan_device_creator.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/common/utils.h"

namespace taichi::lang {
namespace vulkan {

namespace {

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

static const std::unordered_set<std::string> ignored_messages = {
    "UNASSIGNED-DEBUG-PRINTF",
    "VUID_Undefined",
    // (penguinliong): Attempting to map a non-host-visible piece of memory.
    // `VulkanDevice::map()` returns `RhiResult::invalid_usage` in this case.
    "VUID-vkMapMemory-memory-00682",
};

[[maybe_unused]] bool vk_ignore_validation_warning(
    const std::string &msg_name) {
  if (ignored_messages.count(msg_name) > 0) {
    return true;
  }

  return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                  VkDebugUtilsMessageTypeFlagsEXT message_type,
                  const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
                  void *p_user_data) {
  if (message_type == VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT &&
      message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT &&
      strstr(p_callback_data->pMessage, "DEBUG-PRINTF") != nullptr) {
    // Message format is "BLABLA | MessageID=xxxxx | <DEBUG_PRINT_MSG>"
    std::string msg(p_callback_data->pMessage);
    auto const pos = msg.find_last_of("|");
    std::cout << msg.substr(pos + 2);
  }

  if (message_severity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    char msg_buf[4096];
    snprintf(msg_buf, sizeof(msg_buf), "Vulkan validation layer: %d, %s",
             message_type, p_callback_data->pMessage);

    if (is_ci()) {
      auto msg_name = std::string(p_callback_data->pMessageIdName);
      if (!vk_ignore_validation_warning(msg_name))
        TI_ERROR(msg_buf);
    } else {
      RHI_LOG_ERROR(msg_buf);
    }
  }

  return VK_FALSE;
}

void populate_debug_messenger_create_info(
    VkDebugUtilsMessengerCreateInfoEXT *create_info) {
  *create_info = {};
  create_info->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
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

std::vector<const char *> get_required_extensions(bool enable_validation) {
  std::vector<const char *> extensions;
  if (enable_validation) {
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
  for (int i = 0; i < (int)queue_family_count; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if ((masked_flags & VK_QUEUE_COMPUTE_BIT) &&
        !(masked_flags & VK_QUEUE_GRAPHICS_BIT)) {
      indices.compute_family = i;
    }
    if (masked_flags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }

    if (surface != VK_NULL_HANDLE) {
      VkBool32 present_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                           &present_support);
      char msg_buf[128];
      RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf),
                         "Queue %d %s support for presenting", i,
                         present_support ? "has" : "does NOT have");
      RHI_LOG_DEBUG(msg_buf);

      if (present_support) {
        indices.present_family = i;
      }
    }

    if (indices.is_complete() && indices.is_complete_for_ui()) {
      char msg_buf[128];
      RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf),
                         "Found async compute queue %d, graphics queue %d",
                         indices.compute_family.value(),
                         indices.graphics_family.value());
      RHI_LOG_DEBUG(msg_buf);
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

size_t get_device_score(VkPhysicalDevice device, VkSurfaceKHR surface) {
  auto indices = find_queue_families(device, surface);
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(device, &features);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);

  size_t score = 0;

  if (surface != VK_NULL_HANDLE) {
    // this means we need ui
    score = size_t(indices.is_complete_for_ui()) * 1000;
  } else {
    score = size_t(indices.is_complete()) * 1000;
  }

  score += features.wideLines * 100;
  score +=
      size_t(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) *
      500;
  score +=
      size_t(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) *
      1000;
  score += VK_API_VERSION_MINOR(properties.apiVersion) * 100;

  return score;
}

}  // namespace

VulkanDeviceCreator::VulkanDeviceCreator(
    const VulkanDeviceCreator::Params &params)
    : params_(params) {
  if (!VulkanLoader::instance().init()) {
    throw std::runtime_error("Error loading vulkan");
  }

  ti_device_ = std::make_unique<VulkanDevice>();
  uint32_t vk_api_version;
  bool manual_create;
  if (params_.api_version.has_value()) {
    // The version client specified to use
    //
    // If the user provided an API version then the device creation process is
    // totally directed by the information provided externally.
    vk_api_version = params_.api_version.value();
    manual_create = true;
  } else {
    // The highest version designed to use
    vk_api_version = VulkanEnvSettings::k_api_version();
    manual_create = false;
  }

  create_instance(vk_api_version, manual_create);
  setup_debug_messenger();
  VkSurfaceKHR test_surface = VK_NULL_HANDLE;
  if (params_.is_for_ui) {
    test_surface = params_.surface_creator(instance_);
    RHI_ASSERT((test_surface != VK_NULL_HANDLE) &&
               "failed to create window surface!");
  }
  pick_physical_device(test_surface);
  create_logical_device(manual_create);

  {
    VulkanDevice::Params params;
    params.instance = instance_;
    params.physical_device = physical_device_;
    params.device = device_;
    params.compute_queue = compute_queue_;
    params.compute_queue_family_index =
        queue_family_indices_.compute_family.value();
    params.graphics_queue = graphics_queue_;
    params.graphics_queue_family_index =
        queue_family_indices_.graphics_family.value();
    ti_device_->init_vulkan_structs(params);
  }

  if (params_.is_for_ui) {
    vkDestroySurfaceKHR(instance_, test_surface, kNoVkAllocCallbacks);
  }
}

VulkanDeviceCreator::~VulkanDeviceCreator() {
  ti_device_.reset();
  if (params_.enable_validation_layer) {
    destroy_debug_utils_messenger_ext(instance_, debug_messenger_,
                                      kNoVkAllocCallbacks);
  }
  vkDestroyDevice(device_, kNoVkAllocCallbacks);
  vkDestroyInstance(instance_, kNoVkAllocCallbacks);
}

void VulkanDeviceCreator::create_instance(uint32_t vk_api_version,
                                          bool manual_create) {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Taichi Vulkan Backend";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VulkanEnvSettings::k_api_version();

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  if (params_.enable_validation_layer) {
    if (!check_validation_layer_support()) {
      RHI_LOG_ERROR(
          "Validation layers requested but not available, turning off... "
          "Please make sure Vulkan SDK from https://vulkan.lunarg.com/sdk/home "
          "is installed.");
      params_.enable_validation_layer = false;
    }
  }

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  if (params_.enable_validation_layer) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();

    populate_debug_messenger_create_info(&debug_create_info);
    create_info.pNext = &debug_create_info;
  } else {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  // Response to `DebugPrintf`.
  std::array<VkValidationFeatureEnableEXT, 1> vfes = {
      VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  VkValidationFeaturesEXT vf = {};
  if (params_.enable_validation_layer) {
    vf.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    vf.pNext = create_info.pNext;
    vf.enabledValidationFeatureCount = vfes.size();
    vf.pEnabledValidationFeatures = vfes.data();
    create_info.pNext = &vf;
  }

  std::unordered_set<std::string> extensions;
  for (auto &ext : get_required_extensions(params_.enable_validation_layer)) {
    extensions.insert(std::string(ext));
  }
  for (auto &ext : params_.additional_instance_extensions) {
    extensions.insert(std::string(ext));
  }

  uint32_t num_instance_extensions = 0;
  // FIXME: (penguinliong) This was NOT called when `manual_create` is true.
  vkEnumerateInstanceExtensionProperties(nullptr, &num_instance_extensions,
                                         nullptr);
  std::vector<VkExtensionProperties> supported_extensions(
      num_instance_extensions);
  vkEnumerateInstanceExtensionProperties(nullptr, &num_instance_extensions,
                                         supported_extensions.data());

  for (auto &ext : supported_extensions) {
    std::string name = ext.extensionName;
    if (name == VK_KHR_SURFACE_EXTENSION_NAME) {
      extensions.insert(name);
      ti_device_->vk_caps().surface = true;
    } else if (name == VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) {
      extensions.insert(name);
      ti_device_->vk_caps().physical_device_features2 = true;
    } else if (name == VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME) {
      extensions.insert(name);
    } else if (name == VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME) {
      extensions.insert(name);
    } else if (name == VK_EXT_DEBUG_UTILS_EXTENSION_NAME) {
      extensions.insert(name);
    }
  }

  std::vector<const char *> confirmed_extensions;
  confirmed_extensions.reserve(extensions.size());
  for (auto &ext : extensions) {
    confirmed_extensions.push_back(ext.data());
  }

  create_info.enabledExtensionCount = (uint32_t)confirmed_extensions.size();
  create_info.ppEnabledExtensionNames = confirmed_extensions.data();

  VkResult res =
      vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_);

  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkApplicationInfo.html
    // Vulkan 1.0 implementation will return this when api version is not 1.0
    // Vulkan 1.1+ implementation will work with maximum version set
    ti_device_->vk_caps().vk_api_version = VK_API_VERSION_1_0;
    app_info.apiVersion = VK_API_VERSION_1_0;

    res = vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_);
  } else {
    ti_device_->vk_caps().vk_api_version = vk_api_version;
  }

  if (res != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance");
  }

  VulkanLoader::instance().load_instance(instance_);
}

void VulkanDeviceCreator::setup_debug_messenger() {
  if (!params_.enable_validation_layer) {
    return;
  }
  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  populate_debug_messenger_create_info(&create_info);

  BAIL_ON_VK_BAD_RESULT_NO_RETURN(
      create_debug_utils_messenger_ext(instance_, &create_info,
                                       kNoVkAllocCallbacks, &debug_messenger_),
      "failed to set up debug messenger");
}

void VulkanDeviceCreator::pick_physical_device(VkSurfaceKHR test_surface) {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  RHI_ASSERT(device_count > 0 && "failed to find GPUs with Vulkan support");

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
  physical_device_ = VK_NULL_HANDLE;

  for (int i = 0; i < device_count; i++) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(devices[i], &properties);

    char msg_buf[128];
    RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf), "Found Vulkan Device %d (%s)",
                       i, properties.deviceName);
    RHI_LOG_DEBUG(msg_buf);
  }

  auto device_id = VulkanLoader::instance().visible_device_id;
  bool has_visible_device{false};
  if (!device_id.empty()) {
    int id = std::stoi(device_id);
    if (id < 0 || id >= device_count) {
      char msg_buf[128];
      snprintf(msg_buf, sizeof(msg_buf),
               "TI_VISIBLE_DEVICE=%d is not valid, found %d devices available",
               id, device_count);
      RHI_LOG_ERROR(msg_buf);
    } else if (get_device_score(devices[id], test_surface)) {
      physical_device_ = devices[id];
      has_visible_device = true;
    }
  }

  if (!has_visible_device) {
    // could not find a user defined visible device, use the first one suitable
    size_t max_score = 0;
    for (const auto &device : devices) {
      size_t score = get_device_score(device, test_surface);
      if (score > max_score) {
        physical_device_ = device;
        max_score = score;
      }
    }
  }
  RHI_ASSERT(physical_device_ != VK_NULL_HANDLE &&
             "failed to find a suitable GPU");

  queue_family_indices_ = find_queue_families(physical_device_, test_surface);
}

void VulkanDeviceCreator::create_logical_device(bool manual_create) {
  DeviceCapabilityConfig caps{};

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::unordered_set<uint32_t> unique_families;

  if (queue_family_indices_.compute_family.has_value()) {
    unique_families.insert(queue_family_indices_.compute_family.value());
  }
  if (queue_family_indices_.graphics_family.has_value()) {
    unique_families.insert(queue_family_indices_.graphics_family.value());
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
  VkPhysicalDeviceProperties physical_device_properties{};
  vkGetPhysicalDeviceProperties(physical_device_, &physical_device_properties);

  {
    char msg_buf[256];
    RHI_DEBUG_SNPRINTF(
        msg_buf, sizeof(msg_buf),
        "Vulkan Device \"%s\" supports Vulkan %d version %d.%d.%d",
        physical_device_properties.deviceName,
        VK_API_VERSION_VARIANT(physical_device_properties.apiVersion),
        VK_API_VERSION_MAJOR(physical_device_properties.apiVersion),
        VK_API_VERSION_MINOR(physical_device_properties.apiVersion),
        VK_API_VERSION_PATCH(physical_device_properties.apiVersion));
    RHI_LOG_DEBUG(msg_buf);
  }

  // (penguinliong) The actual logical device is created with lastest version of
  // Vulkan but we use the device like it has a lower version (if the user
  // wanted a lower version device).
  uint32_t vk_api_version = physical_device_properties.apiVersion;
  ti_device_->vk_caps().vk_api_version = vk_api_version;
  if (vk_api_version >= VK_API_VERSION_1_3) {
    caps.set(DeviceCapability::spirv_version, 0x10500);
  } else if (vk_api_version >= VK_API_VERSION_1_2) {
    caps.set(DeviceCapability::spirv_version, 0x10500);
  } else if (vk_api_version >= VK_API_VERSION_1_1) {
    caps.set(DeviceCapability::spirv_version, 0x10300);
  } else {
    caps.set(DeviceCapability::spirv_version, 0x10000);
  }

  // Detect extensions
  std::vector<const char *> enabled_extensions;

  uint32_t extension_count = 0;
  // FIXME: (penguinliong) This was NOT called when `manual_create` is true.
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr,
                                       &extension_count, nullptr);
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  vkEnumerateDeviceExtensionProperties(
      physical_device_, nullptr, &extension_count, extension_properties.data());

  bool has_swapchain = false;

  [[maybe_unused]] bool portability_subset_enabled = false;

  for (auto &ext : extension_properties) {
    char msg_buf[256];
    RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf),
                       "Vulkan device extension {%s} (%x)", ext.extensionName,
                       ext.specVersion);
    RHI_LOG_DEBUG(msg_buf);

    std::string name = std::string(ext.extensionName);

    if (name == "VK_KHR_portability_subset") {
      RHI_LOG_ERROR(
          "Potential non-conformant Vulkan implementation, enabling "
          "VK_KHR_portability_subset");
      portability_subset_enabled = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
      has_swapchain = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SPIRV_1_4_EXTENSION_NAME) {
      if (caps.get(DeviceCapability::spirv_version) < 0x10400) {
        caps.set(DeviceCapability::spirv_version, 0x10400);
        enabled_extensions.push_back(ext.extensionName);
      }
    } else if (name == VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME ||
               name == VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) {
      ti_device_->vk_caps().external_memory = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_BIND_MEMORY_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME &&
               params_.enable_validation_layer) {
      // VK_KHR_shader_non_semantic_info isn't supported on molten-vk.
      // Tracking issue: https://github.com/KhronosGroup/MoltenVK/issues/1214
      caps.set(DeviceCapability::spirv_has_non_semantic_info, true);
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_8BIT_STORAGE_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_16BIT_STORAGE_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (std::find(params_.additional_device_extensions.begin(),
                         params_.additional_device_extensions.end(),
                         name) != params_.additional_device_extensions.end()) {
      enabled_extensions.push_back(ext.extensionName);
    }
    // Vulkan doesn't seem to support SPV_KHR_no_integer_wrap_decoration at all.
  }

  if (has_swapchain) {
    ti_device_->vk_caps().present = true;
  }

  VkPhysicalDeviceFeatures device_features{};

  VkPhysicalDeviceFeatures device_supported_features;
  vkGetPhysicalDeviceFeatures(physical_device_, &device_supported_features);

  if (device_supported_features.shaderInt16) {
    device_features.shaderInt16 = true;
    caps.set(DeviceCapability::spirv_has_int16, true);
  }
  if (device_supported_features.shaderInt64) {
    device_features.shaderInt64 = true;
    caps.set(DeviceCapability::spirv_has_int64, true);
  }
  if (device_supported_features.shaderFloat64) {
    device_features.shaderFloat64 = true;
    caps.set(DeviceCapability::spirv_has_float64, true);
  }
  if (device_supported_features.wideLines) {
    device_features.wideLines = true;
    ti_device_->vk_caps().wide_line = true;
  }

  if (ti_device_->vk_caps().vk_api_version >= VK_API_VERSION_1_1) {
    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_properties.pNext = nullptr;

    VkPhysicalDeviceProperties2 physical_device_properties{};
    physical_device_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical_device_properties.pNext = &subgroup_properties;

    vkGetPhysicalDeviceProperties2(physical_device_,
                                   &physical_device_properties);

    if (subgroup_properties.supportedOperations &
        VK_SUBGROUP_FEATURE_BASIC_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_basic, true);
    }
    if (subgroup_properties.supportedOperations &
        VK_SUBGROUP_FEATURE_VOTE_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_vote, true);
    }
    if (subgroup_properties.supportedOperations &
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_arithmetic, true);
    }
    if (subgroup_properties.supportedOperations &
        VK_SUBGROUP_FEATURE_BALLOT_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_ballot, true);
    }
  }

  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount = enabled_extensions.size();
  create_info.ppEnabledExtensionNames = enabled_extensions.data();

  void **pNextEnd = (void **)&create_info.pNext;

  // Use physicalDeviceFeatures2 to features enabled by extensions
  VkPhysicalDeviceVariablePointersFeaturesKHR variable_ptr_feature{};
  variable_ptr_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR;
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shader_atomic_float_feature{};
  shader_atomic_float_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
  VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT shader_atomic_float_2_feature{};
  shader_atomic_float_2_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
  VkPhysicalDeviceFloat16Int8FeaturesKHR shader_f16_i8_feature{};
  shader_f16_i8_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;

  VkPhysicalDevice8BitStorageFeatures shader_8bit_storage_feature{};
  shader_8bit_storage_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
  VkPhysicalDevice16BitStorageFeatures shader_16bit_storage_feature{};
  shader_16bit_storage_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR
      buffer_device_address_feature{};
  buffer_device_address_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature{};
  dynamic_rendering_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;

  if (ti_device_->vk_caps().physical_device_features2) {
    VkPhysicalDeviceFeatures2KHR features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

#define CHECK_EXTENSION(ext)                                          \
  std::find_if(enabled_extensions.begin(), enabled_extensions.end(),  \
               [=](const char *o) { return strcmp(ext, o) == 0; }) != \
      enabled_extensions.end()

    uint32_t vk_api_version = ti_device_->vk_caps().vk_api_version;
#define CHECK_VERSION(major, minor) \
  vk_api_version >= VK_MAKE_API_VERSION(0, major, minor, 0)

    // Variable ptr
    if (CHECK_VERSION(1, 1) ||
        CHECK_EXTENSION(VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME)) {
      features2.pNext = &variable_ptr_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (variable_ptr_feature.variablePointers &&
          variable_ptr_feature.variablePointersStorageBuffer) {
        caps.set(DeviceCapability::spirv_has_variable_ptr, true);
      }
      *pNextEnd = &variable_ptr_feature;
      pNextEnd = &variable_ptr_feature.pNext;
    }

    // Atomic float
    if (CHECK_EXTENSION(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
      features2.pNext = &shader_atomic_float_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);
      if (shader_atomic_float_feature.shaderBufferFloat32AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_atomic_float_add, true);
      }
      if (shader_atomic_float_feature.shaderBufferFloat64AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_atomic_float64_add, true);
      }
      if (shader_atomic_float_feature.shaderBufferFloat32Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_float, true);
      }
      if (shader_atomic_float_feature.shaderBufferFloat64Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_float64, true);
      }
      *pNextEnd = &shader_atomic_float_feature;
      pNextEnd = &shader_atomic_float_feature.pNext;
    }

    // Atomic float 2
    if (CHECK_EXTENSION(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME)) {
      features2.pNext = &shader_atomic_float_2_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);
      if (shader_atomic_float_2_feature.shaderBufferFloat16AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_atomic_float_add, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat16AtomicMinMax) {
        caps.set(DeviceCapability::spirv_has_atomic_float16_minmax, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat16Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_float16, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat32AtomicMinMax) {
        caps.set(DeviceCapability::spirv_has_atomic_float_minmax, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat64AtomicMinMax) {
        caps.set(DeviceCapability::spirv_has_atomic_float64_minmax, true);
      }
      *pNextEnd = &shader_atomic_float_2_feature;
      pNextEnd = &shader_atomic_float_2_feature.pNext;
    }

    // F16 / I8
    if (CHECK_VERSION(1, 2) ||
        CHECK_EXTENSION(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
      features2.pNext = &shader_f16_i8_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (shader_f16_i8_feature.shaderFloat16) {
        caps.set(DeviceCapability::spirv_has_float16, true);
      }
      if (shader_f16_i8_feature.shaderInt8) {
        caps.set(DeviceCapability::spirv_has_int8, true);
      }
      *pNextEnd = &shader_f16_i8_feature;
      pNextEnd = &shader_f16_i8_feature.pNext;
    }

    if (CHECK_VERSION(1, 1) ||
        CHECK_EXTENSION(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
      features2.pNext = &shader_8bit_storage_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      *pNextEnd = &shader_8bit_storage_feature;
      pNextEnd = &shader_8bit_storage_feature.pNext;
    }
    if (CHECK_VERSION(1, 1) ||
        CHECK_EXTENSION(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
      features2.pNext = &shader_16bit_storage_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      *pNextEnd = &shader_16bit_storage_feature;
      pNextEnd = &shader_16bit_storage_feature.pNext;
    }

    // Buffer Device Address
    if (CHECK_VERSION(1, 2) ||
        CHECK_EXTENSION(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
      features2.pNext = &buffer_device_address_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (CHECK_VERSION(1, 3) ||
          buffer_device_address_feature.bufferDeviceAddress) {
        if (device_supported_features.shaderInt64) {
// Temporarily disable it on macOS:
// https://github.com/taichi-dev/taichi/issues/6295
// (penguinliong) Temporarily disabled (until device capability is ready).
#if !defined(__APPLE__) && false
          caps.set(DeviceCapability::spirv_has_physical_storage_buffer, true);
#endif
        }
      }
      *pNextEnd = &buffer_device_address_feature;
      pNextEnd = &buffer_device_address_feature.pNext;
    }

    // Dynamic rendering
    // TODO: Figure out how to integrate this correctly with ImGui,
    //       and then figure out the layout & barrier stuff
    /*
    if (CHECK_EXTENSION(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
      features2.pNext = &dynamic_rendering_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (dynamic_rendering_feature.dynamicRendering) {
        ti_device_->vk_caps().dynamic_rendering = true;
      }

      *pNextEnd = &dynamic_rendering_feature;
      pNextEnd = &dynamic_rendering_feature.pNext;
    }
    */

    // TODO: add atomic min/max feature
  }

  if (params_.enable_validation_layer) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(vkCreateDevice(physical_device_, &create_info,
                                                 kNoVkAllocCallbacks, &device_),
                                  "failed to create logical device");
  VulkanLoader::instance().load_device(device_);

  if (queue_family_indices_.compute_family.has_value()) {
    vkGetDeviceQueue(device_, queue_family_indices_.compute_family.value(), 0,
                     &compute_queue_);
  }
  if (queue_family_indices_.graphics_family.has_value()) {
    vkGetDeviceQueue(device_, queue_family_indices_.graphics_family.value(), 0,
                     &graphics_queue_);
  }

  // Dump capabilities
  caps.dbg_print_all();
  ti_device_->set_caps(std::move(caps));
}

}  // namespace vulkan
}  // namespace taichi::lang
