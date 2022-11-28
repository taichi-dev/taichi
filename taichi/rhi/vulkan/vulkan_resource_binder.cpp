#include "taichi/rhi/vulkan/vulkan_resource_binder.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace vulkan {

VkSampler create_sampler(ImageSamplerConfig config, VkDevice device) {
  VkSampler sampler = VK_NULL_HANDLE;

  // todo: fill these using the information from the ImageSamplerConfig
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
  return sampler;
}

#define CHECK_SET_BINDINGS                                          \
  bool set_not_found = (sets_.find(set) == sets_.end());            \
  if (set_not_found) {                                              \
    if (layout_locked_) {                                           \
      return;                                                       \
    } else {                                                        \
      sets_[set] = {};                                              \
    }                                                               \
  }                                                                 \
  auto &bindings = sets_.at(set).bindings;                          \
  if (layout_locked_ && bindings.find(binding) == bindings.end()) { \
    return;                                                         \
  }

void VulkanResourceBinder::rw_buffer(uint32_t set,
                                     uint32_t binding,
                                     DevicePtr ptr,
                                     size_t size) {
  CHECK_SET_BINDINGS;

  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }

  Binding new_binding = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ptr, size};
  bindings[binding] = new_binding;
}

void VulkanResourceBinder::rw_buffer(uint32_t set,
                                     uint32_t binding,
                                     DeviceAllocation alloc) {
  rw_buffer(set, binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

void VulkanResourceBinder::buffer(uint32_t set,
                                  uint32_t binding,
                                  DevicePtr ptr,
                                  size_t size) {
  CHECK_SET_BINDINGS;

  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }

  Binding new_binding = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ptr, size};
  bindings[binding] = new_binding;
}

void VulkanResourceBinder::buffer(uint32_t set,
                                  uint32_t binding,
                                  DeviceAllocation alloc) {
  buffer(set, binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

void VulkanResourceBinder::image(uint32_t set,
                                 uint32_t binding,
                                 DeviceAllocation alloc,
                                 ImageSamplerConfig sampler_config) {
  CHECK_SET_BINDINGS
  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type ==
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }
  if (bindings[binding].sampler != VK_NULL_HANDLE) {
    Device *dev = bindings[binding].ptr.device;
    vkDestroySampler(static_cast<VulkanDevice *>(dev)->vk_device(),
                     bindings[binding].sampler, kNoVkAllocCallbacks);
  }
  bindings[binding] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       alloc.get_ptr(0), VK_WHOLE_SIZE};
  if (alloc.device) {
    VulkanDevice *device = static_cast<VulkanDevice *>(alloc.device);
    bindings[binding].sampler =
        create_sampler(sampler_config, device->vk_device());
  }
}

void VulkanResourceBinder::rw_image(uint32_t set,
                                    uint32_t binding,
                                    DeviceAllocation alloc,
                                    int lod) {
  CHECK_SET_BINDINGS
  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }
  bindings[binding] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, alloc.get_ptr(0),
                       VK_WHOLE_SIZE};
}

#undef CHECK_SET_BINDINGS

void VulkanResourceBinder::vertex_buffer(DevicePtr ptr, uint32_t binding) {
  vertex_buffers_[binding] = ptr;
}

void VulkanResourceBinder::index_buffer(DevicePtr ptr, size_t index_width) {
  index_buffer_ = ptr;
  if (index_width == 32) {
    index_type_ = VK_INDEX_TYPE_UINT32;
  } else if (index_width == 16) {
    index_type_ = VK_INDEX_TYPE_UINT16;
  } else {
    TI_ERROR("unsupported index width");
  }
}

void VulkanResourceBinder::write_to_set(uint32_t index,
                                        VulkanDevice &device,
                                        vkapi::IVkDescriptorSet set) {
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  std::vector<VkDescriptorImageInfo> image_infos;
  std::vector<bool> is_image;
  std::vector<VkWriteDescriptorSet> desc_writes;

  for (auto &pair : sets_.at(index).bindings) {
    uint32_t binding = pair.first;

    if (pair.second.ptr != kDeviceNullPtr) {
      VkDescriptorBufferInfo &buffer_info = buffer_infos.emplace_back();
      VkDescriptorImageInfo &image_info = image_infos.emplace_back();

      if (pair.second.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pair.second.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
        auto buffer = device.get_vkbuffer(pair.second.ptr);
        buffer_info.buffer = buffer->buffer;
        buffer_info.offset = pair.second.ptr.offset;
        buffer_info.range = pair.second.size;
        is_image.push_back(false);
        set->ref_binding_objs[binding] = buffer;
      } else if (pair.second.type ==
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        auto view = std::get<1>(device.get_vk_image(pair.second.ptr));
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = view->view;
        image_info.sampler = pair.second.sampler;
        is_image.push_back(true);
        set->ref_binding_objs[binding] = view;
      } else if (pair.second.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
        auto view =
            device.get_vk_lod_imageview(pair.second.ptr, pair.second.image_lod);
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_info.imageView = view->view;
        is_image.push_back(true);
        set->ref_binding_objs[binding] = view;
      } else {
        TI_NOT_IMPLEMENTED;
      }

      VkWriteDescriptorSet &write = desc_writes.emplace_back();
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.pNext = nullptr;
      write.dstSet = set->set;
      write.dstBinding = binding;
      write.dstArrayElement = 0;
      write.descriptorCount = 1;
      write.descriptorType = pair.second.type;
      write.pImageInfo = nullptr;
      write.pBufferInfo = nullptr;
      write.pTexelBufferView = nullptr;
    }
  }

  // Set these pointers later as std::vector resize can relocate the pointers
  int i = 0;
  for (auto &write : desc_writes) {
    if (is_image[i]) {
      write.pImageInfo = &image_infos[i];
    } else {
      write.pBufferInfo = &buffer_infos[i];
    }
    i++;
  }

  vkUpdateDescriptorSets(device.vk_device(), desc_writes.size(),
                         desc_writes.data(), /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);
}

void VulkanResourceBinder::lock_layout() {
  layout_locked_ = true;
}

} // namespace vulkan
} // namespace taichi::lang
