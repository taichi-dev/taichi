#include <taichi/rhi/device.h>

namespace taichi::lang {

DeviceAllocationGuard::~DeviceAllocationGuard() {
  device->dealloc_memory(*this);
}

DeviceImageGuard::~DeviceImageGuard() {
  dynamic_cast<GraphicsDevice *>(device)->destroy_image(*this);
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

  return Device::MemcpyCapability::RequiresHost;
}

void Device::memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Intra-device copy
  if (dst.device == src.device) {
    dst.device->memcpy_internal(dst, src, size);
    return;
  }
  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_staging(DevicePtr dst,
                                DevicePtr staging,
                                DevicePtr src,
                                uint64_t size) {
  // Inter-device copy

  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_host(DevicePtr dst,
                             void *host_buffer,
                             DevicePtr src,
                             uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void GraphicsDevice::image_transition(DeviceAllocation img,
                                      ImageLayout old_layout,
                                      ImageLayout new_layout) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->image_transition(img, old_layout, new_layout);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::buffer_to_image(DeviceAllocation dst_img,
                                     DevicePtr src_buf,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->buffer_to_image(dst_img, src_buf, img_layout, params);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::image_to_buffer(DevicePtr dst_buf,
                                     DeviceAllocation src_img,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->image_to_buffer(dst_buf, src_img, img_layout, params);
  stream->submit_synced(cmd_list.get());
}

RhiResult Device::upload_data(DevicePtr *device_ptr,
                              const void **data,
                              size_t *size,
                              int num_alloc) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  std::vector<DeviceAllocationUnique> stagings;
  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [staging, res] = this->allocate_memory_unique(
        {size[i], /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Upload});
    if (res != RhiResult::success) {
      return res;
    }

    void *mapped{nullptr};
    res = this->map(*staging, &mapped);
    if (res != RhiResult::success) {
      return res;
    }
    memcpy(mapped, data[i], size[i]);
    this->unmap(*staging);

    stagings.push_back(std::move(staging));
  }

  Stream *s = this->get_compute_stream();
  auto [cmdlist, res] = s->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }
  for (int i = 0; i < num_alloc; i++) {
    cmdlist->buffer_copy(device_ptr[i], stagings[i]->get_ptr(0), size[i]);
  }
  s->submit_synced(cmdlist.get());

  return RhiResult::success;
}

RhiResult Device::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  Stream *s = this->get_compute_stream();
  auto [cmdlist, res] = s->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }

  std::vector<DeviceAllocationUnique> stagings;
  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [staging, res] = this->allocate_memory_unique(
        {size[i], /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::None});
    if (res != RhiResult::success) {
      return res;
    }

    cmdlist->buffer_copy(staging->get_ptr(0), device_ptr[i], size[i]);
    stagings.push_back(std::move(staging));
  }
  s->submit_synced(cmdlist.get(), wait_sema);

  for (int i = 0; i < num_alloc; i++) {
    void *mapped{nullptr};
    RhiResult res = this->map(*stagings[i], &mapped);
    if (res != RhiResult::success) {
      return res;
    }
    memcpy(data[i], mapped, size[i]);
    this->unmap(*stagings[i]);
  }

  return RhiResult::success;
}

}  // namespace taichi::lang
