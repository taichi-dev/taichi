#pragma once

#include "taichi/backends/device.h"

#include "glad/glad.h"
#include "GLFW/glfw3.h"

namespace taichi {
namespace lang {
namespace opengl {

class GLResourceBinder : public ResourceBinder {
 public:
  ~GLResourceBinder() override;

  struct Bindings {
    // OpenGL has no sets, default set = 0
    uint32_t binding{0};
    GLuint buffer{0};
    GLuint image{0};
  };

  // In Vulkan this is called Storage Buffer (shader can store)
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override;
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override;

  // In Vulkan this is called Uniform Buffer (shader can only load)
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
};

class GLPipeline : public Pipeline {
 public:
  GLPipeline(const PipelineSourceDesc &desc, std::string name);
  ~GLPipeline() override;

  ResourceBinder *resource_binder() override;

 private:
  GLuint program_id_;
};

class GLCommandList : public CommandList {
 public:
  ~GLCommandList() override;

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
};

class GLStream : public Stream {
 public:
  ~GLStream() override;

  std::unique_ptr<CommandList> new_command_list() override;
  void submit(CommandList *cmdlist) override;
  void submit_synced(CommandList *cmdlist) override;

  void command_sync() override;
};

class GLDevice : public GraphicsDevice {
 public:
  ~GLDevice() override;

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  std::unique_ptr<Pipeline> create_pipeline(
      PipelineSourceDesc &src,
      std::string name = "Pipeline") override;

  // Mapping can fail and will return nullptr
  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;

  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;

  // Strictly intra device copy (synced)
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  // Each thraed will acquire its own stream
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
};

class GLSurface : public Surface {
 public:
  ~GLSurface() override;

  DeviceAllocation get_target_image() override;
  void present_image() override;
  std::pair<uint32_t, uint32_t> get_size() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
