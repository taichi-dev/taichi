#pragma once

#include "taichi/backends/device.h"
#include <d3d11.h>

namespace taichi {
namespace lang {
namespace directx11 {

void check_dx_error(HRESULT hr, const char *msg);

HRESULT create_compute_device(ID3D11Device **out_device,
                              ID3D11DeviceContext **out_context,
                              bool force_ref,
                              bool debug_enabled);

class Dx11ResourceBinder : public ResourceBinder {
  ~Dx11ResourceBinder() override;
};

class Dx11Pipeline : public Pipeline {
 public:
  Dx11Pipeline(const PipelineSourceDesc &desc, const std::string &name);
  ~Dx11Pipeline() override;
  ResourceBinder *resource_binder() override;
};

class Dx11Device : public GraphicsDevice {
 public:
  Dx11Device();
  ~Dx11Device() override;

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;
  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override;
  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;
  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;
  Stream *get_compute_stream() override;
  std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") override;
  Stream *get_graphics_stream() override;
  std::unique_ptr<Surface> create_surface(const SurfaceConfig &config) override;
  DeviceAllocation create_image(const ImageParams &params) override;
  void destroy_image(DeviceAllocation handle) override;

  void image_transition(DeviceAllocation img,
                        ImageLayout old_layout,
                        ImageLayout new_layout) override;
  void buffer_to_image(DeviceAllocation dst_img,
                       DevicePtr src_buf,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;
  void image_to_buffer(DevicePtr dst_buf,
                       DeviceAllocation src_img,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;

 private:
  void create_dx11_device();
  void destroy_dx11_device();
  ID3D11Device *device_{};
  ID3D11DeviceContext *context_{};
};

}  // namespace directx11
}  // namespace lang
}  // namespace taichi
