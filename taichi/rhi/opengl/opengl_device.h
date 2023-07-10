#pragma once

#include "taichi/rhi/device.h"

#include "glad/gl.h"
#include "GLFW/glfw3.h"

namespace taichi::lang {
namespace opengl {

class GLDevice;

std::string get_opengl_error_string(GLenum err);

#define check_opengl_error(msg)                                      \
  {                                                                  \
    auto err = glGetError();                                         \
    if (err != GL_NO_ERROR) {                                        \
      auto estr = get_opengl_error_string(err);                      \
      char msgbuf[1024];                                             \
      snprintf(msgbuf, sizeof(msgbuf), "%s: %s", msg, estr.c_str()); \
      RHI_LOG_ERROR(msgbuf);                                         \
      assert(false);                                                 \
    }                                                                \
  }

extern std::optional<void *> kGetOpenglProcAddr;
extern std::optional<void *> imported_process_address;

class GLResourceSet : public ShaderResourceSet {
 public:
  GLResourceSet() = default;
  GLResourceSet(const GLResourceSet &other) = default;

  ~GLResourceSet() override;

  GLResourceSet &rw_buffer(uint32_t binding, DevicePtr ptr, size_t size) final;
  GLResourceSet &rw_buffer(uint32_t binding, DeviceAllocation alloc) final;

  GLResourceSet &buffer(uint32_t binding, DevicePtr ptr, size_t size) final;
  GLResourceSet &buffer(uint32_t binding, DeviceAllocation alloc) final;

  GLResourceSet &image(uint32_t binding,
                       DeviceAllocation alloc,
                       ImageSamplerConfig sampler_config) final;
  GLResourceSet &rw_image(uint32_t binding,
                          DeviceAllocation alloc,
                          int lod) final;

  struct BufferBinding {
    GLuint buffer;
    size_t offset;
    size_t size;
  };

  const std::unordered_map<uint32_t, BufferBinding> &ssbo_binding_map() const {
    return ssbo_binding_map_;
  }

  const std::unordered_map<uint32_t, BufferBinding> &ubo_binding_map() const {
    return ubo_binding_map_;
  }

  const std::unordered_map<uint32_t, GLuint> &texture_binding_map() const {
    return texture_binding_map_;
  }

  const std::unordered_map<uint32_t, GLuint> &rw_image_binding_map() const {
    return rw_image_binding_map_;
  }

 private:
  std::unordered_map<uint32_t, BufferBinding> ssbo_binding_map_;
  std::unordered_map<uint32_t, BufferBinding> ubo_binding_map_;
  std::unordered_map<uint32_t, GLuint> texture_binding_map_;
  std::unordered_map<uint32_t, GLuint> rw_image_binding_map_;
};

class GLPipeline : public Pipeline {
 public:
  GLPipeline(const PipelineSourceDesc &desc, const std::string &name);
  ~GLPipeline() override;

  GLuint get_program() {
    return program_id_;
  }

 private:
  GLuint program_id_;
};

class GLCommandList : public CommandList {
 public:
  explicit GLCommandList(GLDevice *device) : device_(device) {
  }
  ~GLCommandList() override;

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

  // GL only stuff
  void run_commands();

 private:
  struct Cmd {
    virtual void execute() {
    }
    virtual ~Cmd() {
    }
  };

  struct CmdBindPipeline : public Cmd {
    GLuint program{0};
    void execute() override;
  };

  struct CmdBindResources : public Cmd {
    struct BufferBinding {
      GLuint buffer{0};
      GLuint index{0};
      GLuint offset{0};
      GLuint size{0};
      GLenum target{GL_SHADER_STORAGE_BUFFER};
    };

    struct TextureBinding {
      GLuint texture{0};
      GLuint index{0};
      GLenum target{GL_TEXTURE_2D};
      GLenum format{GL_RGBA32F};
      bool is_storage{false};
    };

    std::vector<BufferBinding> buffers;
    std::vector<TextureBinding> textures;
    void execute() override;
  };

  struct CmdBufferBarrier : public Cmd {
    void execute() override;
  };

  struct CmdBufferCopy : public Cmd {
    GLuint src{0}, dst{0};
    size_t src_offset{0}, dst_offset{0};
    size_t size{0};
    void execute() override;
  };

  struct CmdBufferFill : public Cmd {
    GLuint buffer{0};
    size_t offset{0};
    size_t size{0};
    uint32_t data{0};
    void execute() override;
  };

  struct CmdDispatch : public Cmd {
    uint32_t x{0}, y{0}, z{0};
    void execute() override;
  };

  struct CmdImageTransition : public Cmd {
    void execute() override;
  };

  struct CmdBufferToImage : public Cmd {
    BufferImageCopyParams params;
    DeviceAllocationId image{0};
    GLuint buffer{0};
    size_t offset{0};
    GLDevice *device{nullptr};
    void execute() override;
  };

  struct CmdImageToBuffer : public Cmd {
    BufferImageCopyParams params;
    DeviceAllocationId image{0};
    GLuint buffer{0};
    size_t offset{0};
    GLDevice *device{nullptr};
    void execute() override;
  };

  std::vector<std::unique_ptr<Cmd>> recorded_commands_;
  GLDevice *device_{nullptr};
};

class GLStream : public Stream {
 public:
  explicit GLStream(GLDevice *device) : device_(device) {
  }
  ~GLStream() override;

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;

  void command_sync() override;

 private:
  GLDevice *device_{nullptr};
};

struct GLImageAllocation {
  GLenum target;
  GLsizei levels;
  GLenum format;
  GLsizei width;
  GLsizei height;
  GLsizei depth;
  bool external;
};

class GLDevice : public GraphicsDevice {
 public:
  GLDevice();
  ~GLDevice() override;

  Arch arch() const override {
    return Arch::opengl;
  }

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  void dealloc_memory(DeviceAllocation handle) override;

  GLint get_devalloc_size(DeviceAllocation handle);

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

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final;

  ShaderResourceSet *create_resource_set() final {
    return new GLResourceSet;
  }

  RasterResources *create_raster_resources() final {
    TI_NOT_IMPLEMENTED;
  }

  // Mapping can fail and will return nullptr
  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) final;
  void unmap(DeviceAllocation alloc) final;

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

  void wait_idle() override;

  std::unique_ptr<Surface> create_surface(const SurfaceConfig &config) override;
  DeviceAllocation create_image(const ImageParams &params) override;
  void destroy_image(DeviceAllocation handle) override;

  DeviceAllocation import_image(GLuint texture, GLImageAllocation &&gl_image);

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

  GLImageAllocation get_gl_image(GLuint image) const {
    return image_allocs_.at(image);
  }

 private:
  GLStream stream_;
  std::unordered_map<GLuint, GLbitfield> buffer_to_access_;
  std::unordered_map<GLuint, GLImageAllocation> image_allocs_;
};

class GLSurface : public Surface {
 public:
  ~GLSurface() override;

  StreamSemaphore acquire_next_image() override;
  DeviceAllocation get_target_image() override;
  void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  std::pair<uint32_t, uint32_t> get_size() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;
};

}  // namespace opengl
}  // namespace taichi::lang
