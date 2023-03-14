#pragma once

#ifdef TI_WITH_DX11

#include "taichi/rhi/device.h"
#include "taichi/rhi/dx/dx_info_queue.h"
#include <d3d11.h>

namespace taichi::lang {
namespace directx11 {

// Only enable debug layer when the corresponding testing facility is enabled
constexpr bool kD3d11DebugEnabled = false;
// Force REF device. May be used to
// force software rendering.
constexpr bool kD3d11ForceRef = false;
// Enable to spawn a debug window and swapchain
// #define TAICHI_DX11_DEBUG_WINDOW

void check_dx_error(HRESULT hr, const char *msg);

class Dx11ResourceSet : public ShaderResourceSet {
 public:
  Dx11ResourceSet() = default;
  ~Dx11ResourceSet() override;

  ShaderResourceSet &rw_buffer(uint32_t binding,
                               DevicePtr ptr,
                               size_t size) final;
  ShaderResourceSet &rw_buffer(uint32_t binding, DeviceAllocation alloc) final;
  ShaderResourceSet &buffer(uint32_t binding, DevicePtr ptr, size_t size) final;
  ShaderResourceSet &buffer(uint32_t binding, DeviceAllocation alloc) final;
  ShaderResourceSet &image(uint32_t binding,
                           DeviceAllocation alloc,
                           ImageSamplerConfig sampler_config) final;
  ShaderResourceSet &rw_image(uint32_t binding,
                              DeviceAllocation alloc,
                              int lod) final;

  const std::unordered_map<uint32_t, uint32_t> &uav_binding_to_alloc_id() {
    return uav_binding_to_alloc_id_;
  }

  const std::unordered_map<uint32_t, uint32_t> &cb_binding_to_alloc_id() {
    return cb_binding_to_alloc_id_;
  }

 private:
  std::unordered_map<uint32_t, uint32_t> uav_binding_to_alloc_id_;
  std::unordered_map<uint32_t, uint32_t> cb_binding_to_alloc_id_;
};

class Dx11RasterResources : public RasterResources {
  ~Dx11RasterResources() override = default;

  RasterResources &vertex_buffer(DevicePtr ptr, uint32_t binding = 0) final {
    TI_NOT_IMPLEMENTED;
    return *this;
  }

  RasterResources &index_buffer(DevicePtr ptr, size_t index_width) final {
    TI_NOT_IMPLEMENTED;
    return *this;
  }
};

class Dx11Device;

class Dx11Pipeline : public Pipeline {
 public:
  Dx11Pipeline(const PipelineSourceDesc &desc,
               const std::string &name,
               Dx11Device *device);
  ~Dx11Pipeline() override;

  ID3D11ComputeShader *get_program() {
    return compute_shader_;
  }
  const std::string &name() {
    return name_;
  }

 private:
  // Can't use shared_ptr b/c this can cause device_ to be deallocated
  // pre-maturely
  Dx11Device *device_{nullptr};

  ID3D11ComputeShader *compute_shader_{nullptr};
  std::string name_;
};

class Dx11Stream : public Stream {
 public:
  Dx11Stream(Dx11Device *);
  ~Dx11Stream() override;

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  void command_sync() override;

 private:
  Dx11Device *device_{nullptr};
};

class Dx11CommandList : public CommandList {
 public:
  Dx11CommandList(Dx11Device *ti_device);
  ~Dx11CommandList() override;

  void bind_pipeline(Pipeline *p) noexcept final;
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final;
  RhiResult bind_raster_resources(RasterResources *res) noexcept final;
  void buffer_barrier(DevicePtr ptr, size_t size) noexcept final;
  void buffer_barrier(DeviceAllocation alloc) noexcept final;
  void memory_barrier() noexcept final;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept final;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept final;
  RhiResult dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept final;

  // These are not implemented in compute only device
  void begin_renderpass(int x0,
                        int y0,
                        int x1,
                        int y1,
                        uint32_t num_color_attachments,
                        DeviceAllocation *color_attachments,
                        bool *color_clear,
                        std::vector<float> *clear_colors,
                        DeviceAllocation *depth_attachment,
                        bool depth_clear) override;
  void end_renderpass() override;
  void draw(uint32_t num_verticies, uint32_t start_vertex = 0) override;
  void set_line_width(float width) override;
  void draw_indexed(uint32_t num_indicies,
                    uint32_t start_vertex = 0,
                    uint32_t start_index = 0) override;
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

  void run_commands();

 private:
  ID3D11DeviceContext *d3d11_deferred_context_{nullptr};
  ID3D11CommandList *d3d11_command_list_{nullptr};

  std::vector<ID3D11Buffer *> used_spv_workgroup_cb;

  Dx11Device *device_{nullptr};
  int cb_slot_watermark_{-1};
};

class Dx11Device : public GraphicsDevice {
 public:
  Dx11Device();
  ~Dx11Device() override;

  Arch arch() const override {
    return Arch::dx11;
  }

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  void dealloc_memory(DeviceAllocation handle) override;

  RhiResult upload_data(DevicePtr *device_ptr,
                        const void **data,
                        size_t *size,
                        int num_alloc = 1) noexcept final;

  RhiResult readback_data(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc = 1,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept final;

  ShaderResourceSet *create_resource_set() final {
    return new Dx11ResourceSet;
  }

  RasterResources *create_raster_resources() final {
    return new Dx11RasterResources;
  }

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final;
  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;
  void unmap(DevicePtr ptr) final;
  void unmap(DeviceAllocation alloc) final;
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
  void wait_idle() override;

  int live_dx11_object_count();
  ID3D11DeviceContext *d3d11_context() {
    return context_;
  }

  ID3D11Buffer *alloc_id_to_default_copy(uint32_t alloc_id);
  ID3D11Buffer *alloc_id_to_buffer(ID3D11DeviceContext *context,
                                   uint32_t alloc_id);
  ID3D11Buffer *alloc_id_to_staging_buffer(ID3D11DeviceContext *context,
                                           uint32_t alloc_id);
  ID3D11UnorderedAccessView *alloc_id_to_uav(ID3D11DeviceContext *context,
                                             uint32_t alloc_id);
  ID3D11Buffer *alloc_id_to_cb_buffer(ID3D11DeviceContext *context,
                                      uint32_t alloc_id);

  ID3D11Device *d3d11_device() {
    return device_;
  }

  // cb_slot should be 1 after pre-occupied buffers
  // example: in the presence of args_t, cb_slot will be cb0
  // in the absence of args_t, cb_slot will be cb0
  ID3D11Buffer *set_spirv_cross_numworkgroups(uint32_t x,
                                              uint32_t y,
                                              uint32_t z,
                                              int cb_slot);

 private:
  void create_dx11_device();
  void destroy_dx11_device();

  ID3D11Device *device_{nullptr};
  ID3D11DeviceContext *context_{nullptr};
  std::unique_ptr<Dx11InfoQueue> info_queue_{nullptr};

  struct BufferTuple {
    ID3D11Buffer *raw_buffer{nullptr};
    ID3D11Buffer *dynamic_constants{nullptr};
    ID3D11Buffer *staging{nullptr};
    ID3D11UnorderedAccessView *raw_uav{nullptr};

    ID3D11Buffer *mapped{nullptr};

    size_t size{0};
    bool cpu_read{false};
    bool cpu_write{false};
    int default_copy{0};

    ~BufferTuple();

    ID3D11Buffer *get_default_copy(ID3D11Device *device) {
      if (default_copy == 0) {
        return get_raw_buffer(nullptr, device);
      } else if (default_copy == 1) {
        return get_dynamic_constants(nullptr, device);
      } else {
        return get_staging(nullptr, device);
      }
    }

    void clear_derived();

    ID3D11Buffer *get_raw_buffer(ID3D11DeviceContext *context,
                                 ID3D11Device *device);
    ID3D11Buffer *get_dynamic_constants(ID3D11DeviceContext *context,
                                        ID3D11Device *device);
    ID3D11Buffer *get_staging(ID3D11DeviceContext *context,
                              ID3D11Device *device);

    ID3D11Buffer *get_cpu_write_copy(ID3D11DeviceContext *context,
                                     ID3D11Device *device) {
      if (default_copy == 2) {
        return get_staging(context, device);
      }
      return get_dynamic_constants(context, device);
    }

    void copy_back(ID3D11Buffer *buffer,
                   ID3D11DeviceContext *context,
                   ID3D11Device *device);

    ID3D11UnorderedAccessView *get_uav(ID3D11DeviceContext *context,
                                       ID3D11Device *device);
  };

  std::unordered_map<uint32_t, BufferTuple> alloc_id_to_buffer_;
  int alloc_serial_{0};
  std::unique_ptr<Dx11Stream> stream_;
};

}  // namespace directx11
}  // namespace taichi::lang

#endif
