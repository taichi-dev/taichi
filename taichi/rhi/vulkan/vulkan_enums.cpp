#include "taichi/rhi/vulkan/vulkan_enums.h"

namespace taichi::lang {
namespace vulkan {

VkFormat buffer_format_ti_to_vk(BufferFormat f) {
  switch (f) {
    case BufferFormat::r8:
      return VK_FORMAT_R8_UNORM;
    case BufferFormat::rg8:
      return VK_FORMAT_R8G8_UNORM;
    case BufferFormat::rgba8:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case BufferFormat::rgba8srgb:
      return VK_FORMAT_R8G8B8A8_SRGB;
    case BufferFormat::bgra8:
      return VK_FORMAT_B8G8R8A8_UNORM;
    case BufferFormat::bgra8srgb:
      return VK_FORMAT_B8G8R8A8_SRGB;
    case BufferFormat::r8u:
      return VK_FORMAT_R8_UINT;
    case BufferFormat::rg8u:
      return VK_FORMAT_R8G8_UINT;
    case BufferFormat::rgba8u:
      return VK_FORMAT_R8G8B8A8_UINT;
    case BufferFormat::r8i:
      return VK_FORMAT_R8_SINT;
    case BufferFormat::rg8i:
      return VK_FORMAT_R8G8_SINT;
    case BufferFormat::rgba8i:
      return VK_FORMAT_R8G8B8A8_SINT;
    case BufferFormat::r16:
      return VK_FORMAT_R16_UNORM;
    case BufferFormat::rg16:
      return VK_FORMAT_R16G16_UNORM;
    case BufferFormat::rgb16:
      return VK_FORMAT_R16G16B16_UNORM;
    case BufferFormat::rgba16:
      return VK_FORMAT_R16G16B16A16_UNORM;
    case BufferFormat::r16u:
      return VK_FORMAT_R16_UINT;
    case BufferFormat::rg16u:
      return VK_FORMAT_R16G16_UINT;
    case BufferFormat::rgb16u:
      return VK_FORMAT_R16G16B16_UINT;
    case BufferFormat::rgba16u:
      return VK_FORMAT_R16G16B16A16_UINT;
    case BufferFormat::r16i:
      return VK_FORMAT_R16_SINT;
    case BufferFormat::rg16i:
      return VK_FORMAT_R16G16_SINT;
    case BufferFormat::rgb16i:
      return VK_FORMAT_R16G16B16_SINT;
    case BufferFormat::rgba16i:
      return VK_FORMAT_R16G16B16A16_SINT;
    case BufferFormat::r16f:
      return VK_FORMAT_R16_SFLOAT;
    case BufferFormat::rg16f:
      return VK_FORMAT_R16G16_SFLOAT;
    case BufferFormat::rgb16f:
      return VK_FORMAT_R16G16B16_SFLOAT;
    case BufferFormat::rgba16f:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case BufferFormat::r32u:
      return VK_FORMAT_R32_UINT;
    case BufferFormat::rg32u:
      return VK_FORMAT_R32G32_UINT;
    case BufferFormat::rgb32u:
      return VK_FORMAT_R32G32B32_UINT;
    case BufferFormat::rgba32u:
      return VK_FORMAT_R32G32B32A32_UINT;
    case BufferFormat::r32i:
      return VK_FORMAT_R32_SINT;
    case BufferFormat::rg32i:
      return VK_FORMAT_R32G32_SINT;
    case BufferFormat::rgb32i:
      return VK_FORMAT_R32G32B32_SINT;
    case BufferFormat::rgba32i:
      return VK_FORMAT_R32G32B32A32_SINT;
    case BufferFormat::r32f:
      return VK_FORMAT_R32_SFLOAT;
    case BufferFormat::rg32f:
      return VK_FORMAT_R32G32_SFLOAT;
    case BufferFormat::rgb32f:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case BufferFormat::rgba32f:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case BufferFormat::depth16:
      return VK_FORMAT_D16_UNORM;
    case BufferFormat::depth24stencil8:
      return VK_FORMAT_D24_UNORM_S8_UINT;
    case BufferFormat::depth32f:
      return VK_FORMAT_D32_SFLOAT;
    default:
      TI_ERROR("BufferFormat cannot be mapped to VkFormat");
      return VK_FORMAT_UNDEFINED;
  }
}

BufferFormat buffer_format_vk_to_ti(VkFormat f) {
  switch (f) {
    case VK_FORMAT_R8_UNORM:
      return BufferFormat::r8;
    case VK_FORMAT_R8G8_UNORM:
      return BufferFormat::rg8;
    case VK_FORMAT_R8G8B8A8_UNORM:
      return BufferFormat::rgba8;
    case VK_FORMAT_R8G8B8A8_SRGB:
      return BufferFormat::rgba8srgb;
    case VK_FORMAT_B8G8R8A8_UNORM:
      return BufferFormat::bgra8;
    case VK_FORMAT_B8G8R8A8_SRGB:
      return BufferFormat::bgra8srgb;
    case VK_FORMAT_R8_UINT:
      return BufferFormat::r8u;
    case VK_FORMAT_R8G8_UINT:
      return BufferFormat::rg8u;
    case VK_FORMAT_R8G8B8A8_UINT:
      return BufferFormat::rgba8u;
    case VK_FORMAT_R8_SINT:
      return BufferFormat::r8i;
    case VK_FORMAT_R8G8_SINT:
      return BufferFormat::rg8i;
    case VK_FORMAT_R8G8B8A8_SINT:
      return BufferFormat::rgba8i;
    case VK_FORMAT_R16_UNORM:
      return BufferFormat::r16;
    case VK_FORMAT_R16G16_UNORM:
      return BufferFormat::rg16;
    case VK_FORMAT_R16G16B16_UNORM:
      return BufferFormat::rgb16;
    case VK_FORMAT_R16G16B16A16_UNORM:
      return BufferFormat::rgba16;
    case VK_FORMAT_R16_UINT:
      return BufferFormat::r16u;
    case VK_FORMAT_R16G16_UINT:
      return BufferFormat::rg16u;
    case VK_FORMAT_R16G16B16_UINT:
      return BufferFormat::rgb16u;
    case VK_FORMAT_R16G16B16A16_UINT:
      return BufferFormat::rgba16u;
    case VK_FORMAT_R16_SINT:
      return BufferFormat::r16i;
    case VK_FORMAT_R16G16_SINT:
      return BufferFormat::rg16i;
    case VK_FORMAT_R16G16B16_SINT:
      return BufferFormat::rgb16i;
    case VK_FORMAT_R16G16B16A16_SINT:
      return BufferFormat::rgba16i;
    case VK_FORMAT_R16_SFLOAT:
      return BufferFormat::r16f;
    case VK_FORMAT_R16G16_SFLOAT:
      return BufferFormat::rg16f;
    case VK_FORMAT_R16G16B16_SFLOAT:
      return BufferFormat::rgb16f;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return BufferFormat::rgba16f;
    case VK_FORMAT_R32_UINT:
      return BufferFormat::r32u;
    case VK_FORMAT_R32G32_UINT:
      return BufferFormat::rg32u;
    case VK_FORMAT_R32G32B32_UINT:
      return BufferFormat::rgb32u;
    case VK_FORMAT_R32G32B32A32_UINT:
      return BufferFormat::rgba32u;
    case VK_FORMAT_R32_SINT:
      return BufferFormat::r32i;
    case VK_FORMAT_R32G32_SINT:
      return BufferFormat::rg32i;
    case VK_FORMAT_R32G32B32_SINT:
      return BufferFormat::rgb32i;
    case VK_FORMAT_R32G32B32A32_SINT:
      return BufferFormat::rgba32i;
    case VK_FORMAT_R32_SFLOAT:
      return BufferFormat::r32f;
    case VK_FORMAT_R32G32_SFLOAT:
      return BufferFormat::rg32f;
    case VK_FORMAT_R32G32B32_SFLOAT:
      return BufferFormat::rgb32f;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return BufferFormat::rgba32f;
    case VK_FORMAT_D16_UNORM:
      return BufferFormat::depth16;
    case VK_FORMAT_D24_UNORM_S8_UINT:
      return BufferFormat::depth24stencil8;
    case VK_FORMAT_D32_SFLOAT:
      return BufferFormat::depth32f;
    default:
      TI_ERROR("VkFormat cannot be mapped to BufferFormat");
      return BufferFormat::unknown;
  }
}

VkImageLayout image_layout_ti_to_vk(ImageLayout layout) {
  switch (layout) {
    case ImageLayout::undefined:
      return VK_IMAGE_LAYOUT_UNDEFINED;
    case ImageLayout::shader_read:
      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case ImageLayout::shader_write:
      return VK_IMAGE_LAYOUT_GENERAL;
    case ImageLayout::shader_read_write:
      return VK_IMAGE_LAYOUT_GENERAL;
    case ImageLayout::color_attachment:
      return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case ImageLayout::color_attachment_read:
      return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case ImageLayout::depth_attachment:
      return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    case ImageLayout::depth_attachment_read:
      return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    case ImageLayout::transfer_dst:
      return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case ImageLayout::transfer_src:
      return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case ImageLayout::present_src:
      return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    default:
      TI_ERROR("ImageLayout cannot be mapped to VkImageLayout");
      return VK_IMAGE_LAYOUT_UNDEFINED;
  }
}

VkBlendOp blend_op_ti_to_vk(BlendOp op) {
  switch (op) {
    case BlendOp::add:
      return VK_BLEND_OP_ADD;
    case BlendOp::subtract:
      return VK_BLEND_OP_SUBTRACT;
    case BlendOp::reverse_subtract:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    case BlendOp::min:
      return VK_BLEND_OP_MIN;
    case BlendOp::max:
      return VK_BLEND_OP_MAX;
    default:
      TI_ERROR("BlendOp cannot be mapped to VkBlendOp");
      return VK_BLEND_OP_ADD;
  }
}

VkBlendFactor blend_factor_ti_to_vk(BlendFactor factor) {
  switch (factor) {
    case BlendFactor::zero:
      return VK_BLEND_FACTOR_ZERO;
    case BlendFactor::one:
      return VK_BLEND_FACTOR_ONE;
    case BlendFactor::src_color:
      return VK_BLEND_FACTOR_SRC_COLOR;
    case BlendFactor::one_minus_src_color:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case BlendFactor::dst_color:
      return VK_BLEND_FACTOR_DST_COLOR;
    case BlendFactor::one_minus_dst_color:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case BlendFactor::src_alpha:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case BlendFactor::one_minus_src_alpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case BlendFactor::dst_alpha:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case BlendFactor::one_minus_dst_alpha:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    default:
      TI_ERROR("BlendFactor cannot be mapped to VkBlendFactor");
      return VK_BLEND_FACTOR_ZERO;
  }
}

}  // namespace vulkan
}  // namespace taichi::lang
