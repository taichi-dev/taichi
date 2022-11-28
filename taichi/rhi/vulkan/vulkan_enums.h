#pragma once
#include "taichi/rhi/vulkan/vulkan_common.h"

namespace taichi::lang {
namespace vulkan {

VkFormat buffer_format_ti_to_vk(BufferFormat f);
BufferFormat buffer_format_vk_to_ti(VkFormat f);

VkImageLayout image_layout_ti_to_vk(ImageLayout layout);

VkBlendOp blend_op_ti_to_vk(BlendOp op);

VkBlendFactor blend_factor_ti_to_vk(BlendFactor factor);

}  // namespace vulkan
}  // namespace taichi::lang
