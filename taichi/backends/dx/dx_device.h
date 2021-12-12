#pragma once

#include "taichi/backends/device.h"
#include <d3d11.h>

namespace taichi {
namespace lang {
namespace directx11 {

class DxDevice;

class DxResourceBinder : public ResourceBinder {
 public:
  ~DxResourceBinder() override;
  std::unique_ptr<ResourceBinder::Bindings> materialize() override;
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override;
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override;
  void buffer(uint32_t set,
              uint32_t binding,
              DevicePtr ptr,
              size_t size) override;
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) override;
  void image(uint32_t set,
             uint32_t binding,
             DeviceAllocation alloc,
             ImageSamplerConfig sampler_config) override;

  // Set vertex buffer (not implemented in compute only device)
  void vertex_buffer(DevicePtr ptr, uint32_t binding = 0) override;

  // Set index buffer (not implemented in compute only device)
  // index_width = 4 -> uint32 index
  // index_width = 2 -> uint16 index
  void index_buffer(DevicePtr ptr, size_t index_width) override;

  const std::unordered_map<uint32_t, uint32_t> &binding_to_alloc_id() {
    return binding_to_alloc_id_;
  }

 private:
  std::unordered_map<uint32_t, uint32_t> binding_to_alloc_id_;
};

class DxPipeline : public Pipeline {
 public:
  DxPipeline(const PipelineSourceDesc &desc, const std::string &name);
  ~DxPipeline() override;
  ResourceBinder *resource_binder() override;
  ID3D11ComputeShader *get_program() {
    return compute_shader_;
  }

 private:
  ID3D11ComputeShader *compute_shader_;
  DxResourceBinder binder_;
};

class DxCommandList : public CommandList {
 public:
  DxCommandList(DxDevice *ti_device);
  ~DxCommandList() override;

  void bind_pipeline(Pipeline *p) override;
  void bind_resources(ResourceBinder *binder) override;
  void bind_resources(ResourceBinder *binder,
                      ResourceBinder::Bindings *bindings) override;
  void buffer_barrier(DevicePtr ptr, size_t size) override;
  void buffer_barrier(DeviceAllocation alloc) override;
  void memory_barrier() override;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override;
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;

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
  void clear_color(float r, float g, float b, float a) override;
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
  struct Cmd {  // Dx-specific CMDs I guess
    virtual void execute() {
    }
  };

  struct CmdBindPipeline : public Cmd {
    ID3D11ComputeShader *compute_shader_;
    void execute() override;
  };

  struct CmdBufferFill : public Cmd {
    // ID3D11Buffer *buffer = nullptr;
    ID3D11UnorderedAccessView *uav;
    size_t offset{0}, size{0};
    uint32_t data{0};
    void execute() override;
  };

  struct CmdBufferCopy : public Cmd {
    ID3D11Buffer *dst, *src;
    size_t src_offset{0}, dst_offset{0}, size{0};
    void execute() override;
  };

  struct CmdBindBufferToIndex : public Cmd {
    ID3D11UnorderedAccessView *uav;  // UAV of the buffer
    uint32_t binding;                // U register; UAV slot
    void execute() override;
  };

  struct CmdDispatch : public Cmd {
    uint32_t x{0}, y{0}, z{0};
    void execute() override;
  };

  std::vector<std::unique_ptr<Cmd>> recorded_commands_;
  DxDevice *device_;
};

class DxStream : public Stream {
 public:
  DxStream(DxDevice *);
  ~DxStream() override;
  std::unique_ptr<CommandList> new_command_list() override;
  void submit(CommandList *cmdlist) override;
  void submit_synced(CommandList *cmdlist) override;
  void command_sync() override;

 private:
  DxDevice *device_;
};

class DxDevice : public GraphicsDevice {
 public:
  DxDevice();
  ~DxDevice() override;

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

  ID3D11Buffer *alloc_id_to_buffer(uint32_t alloc_id);
  ID3D11Buffer *alloc_id_to_buffer_cpu_copy(uint32_t alloc_id);
  ID3D11UnorderedAccessView *alloc_id_to_uav(uint32_t alloc_id);

  static ID3D11Device *device_;
  static ID3D11DeviceContext *context_;
  static void create_dx11_device();

 private:
  DxStream *stream_;
  std::unordered_map<uint32_t, ID3D11Buffer *>
      alloc_id_to_buffer_;  // binding ID to buffer
  std::unordered_map<uint32_t, ID3D11Buffer *>
      alloc_id_to_cpucopy_;  // binding ID to CPU copy of buffer
  std::unordered_map<uint32_t, ID3D11UnorderedAccessView *>
      alloc_id_to_uav_;  // binding ID to UAV
  int alloc_serial_;
};

}  // namespace directx11
}  // namespace lang
}  // namespace taichi