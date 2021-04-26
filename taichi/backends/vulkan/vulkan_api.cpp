#include "taichi/backends/vulkan/vulkan_api.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef IN_TAI_VULKAN

#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/common/logging.h"

#else

#include "vulkan_common_stub.h"

#endif

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
    std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;
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

VulkanQueueFamilyIndices find_queue_families(VkPhysicalDevice device) {
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
    if (indices.is_complete()) {
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

bool is_device_suitable(VkPhysicalDevice device) {
  return find_queue_families(device).is_complete();
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

VulkanDevice::VulkanDevice(const Params &params) {
  create_instance(params);
  setup_debug_messenger();
  pick_physical_device();
  create_logical_device();
  create_command_pool();
}

VulkanDevice::~VulkanDevice() {
  if constexpr (kEnableValidationLayers) {
    destroy_debug_utils_messenger_ext(instance_, debug_messenger_,
                                      kNoVkAllocCallbacks);
  }
  vkDestroyCommandPool(device_, command_pool_, kNoVkAllocCallbacks);
  vkDestroyDevice(device_, kNoVkAllocCallbacks);
  vkDestroyInstance(instance_, kNoVkAllocCallbacks);
}

void VulkanDevice::create_instance(const Params &params) {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Taichi Vulkan Backend";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = params.api_version;  // important

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
  const auto extensions = get_required_extensions();
  create_info.enabledExtensionCount = (uint32_t)extensions.size();
  create_info.ppEnabledExtensionNames = extensions.data();

  BAIL_ON_VK_BAD_RESULT(
      vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_),
      "failed to create instance");
}

void VulkanDevice::setup_debug_messenger() {
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

void VulkanDevice::pick_physical_device() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  TI_ASSERT_INFO(device_count > 0, "failed to find GPUs with Vulkan support");

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
  physical_device_ = VK_NULL_HANDLE;
  for (const auto &device : devices) {
    if (is_device_suitable(device)) {
      physical_device_ = device;
      break;
    }
  }
  TI_ASSERT_INFO(physical_device_ != VK_NULL_HANDLE,
                 "failed to find a suitable GPU");

  queue_family_indices_ = find_queue_families(physical_device_);
}

void VulkanDevice::create_logical_device() {
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex =
      queue_family_indices_.compute_family.value();
  queue_create_info.queueCount = 1;
  constexpr float kQueuePriority = 1.0f;
  queue_create_info.pQueuePriorities = &kQueuePriority;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = &queue_create_info;
  create_info.queueCreateInfoCount = 1;

  VkPhysicalDeviceFeatures device_deatures{};
  create_info.pEnabledFeatures = &device_deatures;
  create_info.enabledExtensionCount = 0;

  if constexpr (kEnableValidationLayers) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }
  BAIL_ON_VK_BAD_RESULT(vkCreateDevice(physical_device_, &create_info,
                                       kNoVkAllocCallbacks, &device_),
                        "failed to create logical device");
  vkGetDeviceQueue(device_, queue_family_indices_.compute_family.value(),
                   /*queueIndex=*/0, &compute_queue_);
}

void VulkanDevice::create_command_pool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = 0;
  pool_info.queueFamilyIndex = queue_family_indices_.compute_family.value();
  BAIL_ON_VK_BAD_RESULT(
      vkCreateCommandPool(device_, &pool_info, kNoVkAllocCallbacks,
                          &command_pool_),
      "failed to create command pool");
}

VulkanPipeline::VulkanPipeline(const Params &params)
    : device_(params.device->device()) {
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

void VulkanCommandBuilder::append(const VulkanPipeline &pipeline,
                                  int group_count_x) {
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
}

VkCommandBuffer VulkanCommandBuilder::build() {
  BAIL_ON_VK_BAD_RESULT(vkEndCommandBuffer(command_buffer_),
                        "failed to record command buffer");
  VkCommandBuffer res = command_buffer_;
  command_buffer_ = VK_NULL_HANDLE;
  return res;
}

VkCommandBuffer record_copy_buffer_command(
    const VulkanDevice *device,
    VkBuffer src_buffer,
    VkBuffer dst_buffer,
    VkDeviceSize size,
    VulkanCopyBufferDirection direction) {
  VkCommandBuffer command{VK_NULL_HANDLE};
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = device->command_pool();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  BAIL_ON_VK_BAD_RESULT(
      vkAllocateCommandBuffers(device->device(), &alloc_info, &command),
      "failed to allocate copy command buffer");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = 0;
  begin_info.pInheritanceInfo = nullptr;
  BAIL_ON_VK_BAD_RESULT(vkBeginCommandBuffer(command, &begin_info),
                        "failed to begin recording copy command buffer");

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;
  vkCmdCopyBuffer(command, src_buffer, dst_buffer, /*regionCount=*/1,
                  &copy_region);
  if (direction == VulkanCopyBufferDirection::H2D) {
    VkMemoryBarrier barrier_info;
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;

    barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_info.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command,
                         /*srcStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT,
                         /*dstStageMask=*/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier_info, 0, nullptr, 0, nullptr);
  }
  BAIL_ON_VK_BAD_RESULT(vkEndCommandBuffer(command),
                        "failed to record copy command buffer");
  return command;
}

VulkanStream::VulkanStream(const VulkanDevice *device) : device_(device) {
}

void VulkanStream::launch(VkCommandBuffer command) {
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command;

  BAIL_ON_VK_BAD_RESULT(
      vkQueueSubmit(device_->compute_queue(), /*submitCount=*/1, &submit_info,
                    /*fence=*/VK_NULL_HANDLE),
      "failed to submit command buffer");
}

void VulkanStream::synchronize() {
  // While vkQueueWaitIdle is strongly discouraged, this is probably the most
  // viable way for synchronization in Taichi. Unlike graphics pipeline, there
  // is no clear boundary (i.e. frame) for us to use a VkFence. TVM accumulates
  // all the commands into a single buffer, then submits it all at once upon
  // synchronization. Not sure how efficient that model is.
  vkQueueWaitIdle(device_->compute_queue());
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
