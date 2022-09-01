#include <taichi/rhi/device.h>

#if TI_WITH_VULKAN
#include <taichi/rhi/vulkan/vulkan_device.h>
#include <taichi/rhi/interop/vulkan_cpu_interop.h>
#if TI_WITH_LLVM
#include <taichi/rhi/cpu/cpu_device.h>
#endif
#if TI_WITH_CUDA
#include <taichi/rhi/cuda/cuda_device.h>
#include <taichi/rhi/interop/vulkan_cuda_interop.h>
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN

namespace taichi {
namespace lang {

// FIXME: (penguinliong) We might have to differentiate buffer formats and
// texture formats at some point because formats like `rgb10a2` are not easily
// represented by primitive types.
std::pair<DataType, uint32_t> buffer_format2type_channels(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 1);
    case BufferFormat::rg8:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 2);
    case BufferFormat::rgba8:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 4);
    case BufferFormat::rgba8srgb:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 4);
    case BufferFormat::bgra8:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 4);
    case BufferFormat::bgra8srgb:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 4);
    case BufferFormat::r8u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 1);
    case BufferFormat::rg8u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 2);
    case BufferFormat::rgba8u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u8), 4);
    case BufferFormat::r8i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i8), 1);
    case BufferFormat::rg8i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i8), 2);
    case BufferFormat::rgba8i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i8), 4);
    case BufferFormat::r16:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 1);
    case BufferFormat::rg16:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 2);
    case BufferFormat::rgb16:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 3);
    case BufferFormat::rgba16:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 4);
    case BufferFormat::r16u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 1);
    case BufferFormat::rg16u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 2);
    case BufferFormat::rgb16u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 3);
    case BufferFormat::rgba16u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u16), 4);
    case BufferFormat::r16i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i16), 1);
    case BufferFormat::rg16i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i16), 2);
    case BufferFormat::rgb16i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i16), 3);
    case BufferFormat::rgba16i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i16), 4);
    case BufferFormat::r16f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f16), 1);
    case BufferFormat::rg16f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f16), 2);
    case BufferFormat::rgb16f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f16), 3);
    case BufferFormat::rgba16f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f16), 4);
    case BufferFormat::r32u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u32), 1);
    case BufferFormat::rg32u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u32), 2);
    case BufferFormat::rgb32u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u32), 3);
    case BufferFormat::rgba32u:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::u32), 4);
    case BufferFormat::r32i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i32), 1);
    case BufferFormat::rg32i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i32), 2);
    case BufferFormat::rgb32i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i32), 3);
    case BufferFormat::rgba32i:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::i32), 4);
    case BufferFormat::r32f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f32), 1);
    case BufferFormat::rg32f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f32), 2);
    case BufferFormat::rgb32f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f32), 3);
    case BufferFormat::rgba32f:
      return std::make_pair<DataType, uint32_t>(
          PrimitiveType::get(PrimitiveTypeID::f32), 4);
    default:
      TI_ERROR("Invalid buffer format");
      return {};
  }
}
BufferFormat type_channels2buffer_format(const DataType &type,
                                         uint32_t num_channels) {
  BufferFormat format;
  if (type == PrimitiveType::f16) {
    if (num_channels == 1) {
      format = BufferFormat::r16f;
    } else if (num_channels == 2) {
      format = BufferFormat::rg16f;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba16f;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else if (type == PrimitiveType::u16) {
    if (num_channels == 1) {
      format = BufferFormat::r16;
    } else if (num_channels == 2) {
      format = BufferFormat::rg16;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba16;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else if (type == PrimitiveType::u8) {
    if (num_channels == 1) {
      format = BufferFormat::r8;
    } else if (num_channels == 2) {
      format = BufferFormat::rg8;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba8;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else if (type == PrimitiveType::f32) {
    if (num_channels == 1) {
      format = BufferFormat::r32f;
    } else if (num_channels == 2) {
      format = BufferFormat::rg32f;
    } else if (num_channels == 3) {
      format = BufferFormat::rgb32f;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba32f;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else {
    TI_ERROR("Invalid texture dtype");
  }
  return format;
}

DeviceAllocationGuard::~DeviceAllocationGuard() {
  device->dealloc_memory(*this);
}

DevicePtr DeviceAllocation::get_ptr(uint64_t offset) const {
  return DevicePtr{{device, alloc_id}, offset};
}

Device::MemcpyCapability Device::check_memcpy_capability(DevicePtr dst,
                                                         DevicePtr src,
                                                         uint64_t size) {
  if (dst.device == src.device) {
    return Device::MemcpyCapability::Direct;
  }

#if TI_WITH_VULKAN
#if TI_WITH_LLVM
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
    // TODO: support direct copy if dst itself supports host write.
    return Device::MemcpyCapability::RequiresStagingBuffer;
  } else if (dynamic_cast<cpu::CpuDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    return Device::MemcpyCapability::RequiresStagingBuffer;
  }
#endif
#if TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
    // FIXME: direct copy isn't always possible.
    // The vulkan buffer needs export_sharing turned on.
    // Otherwise, needs staging buffer
    return Device::MemcpyCapability::Direct;
  } else if (dynamic_cast<cuda::CudaDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    return Device::MemcpyCapability::Direct;
  }
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN
  return Device::MemcpyCapability::RequiresHost;
}

void Device::memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Intra-device copy
  if (dst.device == src.device) {
    dst.device->memcpy_internal(dst, src, size);
    return;
  }
  // Inter-device copy
#if TI_WITH_VULKAN && TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
    memcpy_cuda_to_vulkan(dst, src, size);
    return;
  } else if (dynamic_cast<cuda::CudaDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    memcpy_vulkan_to_cuda(dst, src, size);
    return;
  }
#endif
  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_staging(DevicePtr dst,
                                DevicePtr staging,
                                DevicePtr src,
                                uint64_t size) {
  // Inter-device copy
#if defined(TI_WITH_VULKAN) && defined(TI_WITH_LLVM)
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
    memcpy_cpu_to_vulkan_via_staging(dst, staging, src, size);
    return;
  }
#endif

  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_host(DevicePtr dst,
                             void *host_buffer,
                             DevicePtr src,
                             uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

const std::string to_string(DeviceCapability c) {
#define PER_DEVICE_CAPABILITY(name) \
  case DeviceCapability::name:      \
    return #name;                   \
    break;
  switch (c) {
#include "taichi/inc/rhi_constants.inc.h"
    default:
      return "Unknown";
      break;
  }
#undef PER_DEVICE_CAPABILITY
}

void Device::print_all_cap() const {
  for (auto &pair : caps_) {
    TI_TRACE("DeviceCapability::{} ({}) = {}", to_string(pair.first),
             int(pair.first), pair.second);
  }
}

void GraphicsDevice::image_transition(DeviceAllocation img,
                                      ImageLayout old_layout,
                                      ImageLayout new_layout) {
  Stream *stream = get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->image_transition(img, old_layout, new_layout);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::buffer_to_image(DeviceAllocation dst_img,
                                     DevicePtr src_buf,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->buffer_to_image(dst_img, src_buf, img_layout, params);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::image_to_buffer(DevicePtr dst_buf,
                                     DeviceAllocation src_img,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->image_to_buffer(dst_buf, src_img, img_layout, params);
  stream->submit_synced(cmd_list.get());
}

}  // namespace lang
}  // namespace taichi
