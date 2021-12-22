#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {

Dx11ResourceBinder::~Dx11ResourceBinder() {
}

Dx11Pipeline::Dx11Pipeline(const PipelineSourceDesc &desc,
                           const std::string &name) {
  TI_NOT_IMPLEMENTED;
}

Dx11Pipeline::~Dx11Pipeline() {
}

ResourceBinder *Dx11Pipeline::resource_binder() {
  return nullptr;
}

Dx11Device::Dx11Device() {
  set_cap(DeviceCapability::spirv_version, 0x10300);
}

Dx11Device::~Dx11Device() {
}

DeviceAllocation Dx11Device::allocate_memory(const AllocParams &params) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::dealloc_memory(DeviceAllocation handle) {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Pipeline> Dx11Device::create_pipeline(
    const PipelineSourceDesc &src,
    std::string name) {
  TI_NOT_IMPLEMENTED;
}

void *Dx11Device::map_range(DevicePtr ptr, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void *Dx11Device::map(DeviceAllocation alloc) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::unmap(DevicePtr ptr) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::unmap(DeviceAllocation alloc) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

Stream *Dx11Device::get_compute_stream() {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Pipeline> Dx11Device::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  TI_NOT_IMPLEMENTED;
}

Stream *Dx11Device::get_graphics_stream() {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Surface> Dx11Device::create_surface(const SurfaceConfig &config) {
  TI_NOT_IMPLEMENTED;
}

DeviceAllocation Dx11Device::create_image(const ImageParams &params) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::destroy_image(DeviceAllocation handle) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}
void Dx11Device::image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi