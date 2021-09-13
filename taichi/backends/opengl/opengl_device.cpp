#include "opengl_device.h"

namespace taichi {
namespace lang {
namespace opengl {

GLResourceBinder::~GLResourceBinder() {
}

void GLResourceBinder::rw_buffer(uint32_t set,
                                 uint32_t binding,
                                 DevicePtr ptr,
                                 size_t size) {
}

void GLResourceBinder::rw_buffer(uint32_t set,
                                 uint32_t binding,
                                 DeviceAllocation alloc) {
}

void GLResourceBinder::buffer(uint32_t set,
                              uint32_t binding,
                              DevicePtr ptr,
                              size_t size) {
}

void GLResourceBinder::buffer(uint32_t set,
                              uint32_t binding,
                              DeviceAllocation alloc) {
}

void GLResourceBinder::image(uint32_t set,
                             uint32_t binding,
                             DeviceAllocation alloc,
                             ImageSamplerConfig sampler_config) {
  TI_NOT_IMPLEMENTED;
}

void GLResourceBinder::vertex_buffer(DevicePtr ptr, uint32_t binding) {
  TI_NOT_IMPLEMENTED;
}

void GLResourceBinder::index_buffer(DevicePtr ptr, size_t index_width) {
  TI_NOT_IMPLEMENTED;
}

GLPipeline::GLPipeline(const PipelineSourceDesc &desc, std::string name) {
  TI_ASSERT(desc.type == PipelineSourceType::glsl_src);

  GLuint shader_id;
  shader_id = glCreateShader(GL_COMPUTE_SHADER);

  const GLchar *source_cstr = (const GLchar *)desc.data;
  glShaderSource(shader_id, 1, &source_cstr, nullptr);

  glCompileShader(shader_id);
  int status = GL_TRUE;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    GLsizei logLength;
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &logLength);
    auto log = std::vector<GLchar>(logLength + 1);
    glGetShaderInfoLog(shader_id, logLength, &logLength, log.data());
    log[logLength] = 0;
    TI_ERROR("[glsl] error while compiling shader:\n{}", log.data());
  }

  program_id_ = glCreateProgram();
  glAttachShader(program_id_, shader_id);
  glLinkProgram(program_id_);
  glGetProgramiv(program_id_, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    GLsizei logLength;
    glGetProgramiv(program_id_, GL_INFO_LOG_LENGTH, &logLength);
    auto log = std::vector<GLchar>(logLength + 1);
    glGetProgramInfoLog(program_id_, logLength, &logLength, log.data());
    log[logLength] = 0;
    TI_ERROR("[glsl] error while linking program:\n{}", log.data());
  }

  glDeleteShader(shader_id);
}

GLPipeline::~GLPipeline() {
  glDeleteProgram(program_id_);
}

ResourceBinder *GLPipeline::resource_binder() {
  return nullptr;
}

GLCommandList::~GLCommandList() {
}

void GLCommandList::bind_pipeline(Pipeline *p) {
}

void GLCommandList::bind_resources(ResourceBinder *binder) {
}

void GLCommandList::bind_resources(ResourceBinder *binder,
                                   ResourceBinder::Bindings *bindings) {
}

void GLCommandList::buffer_barrier(DevicePtr ptr, size_t size) {
}

void GLCommandList::buffer_barrier(DeviceAllocation alloc) {
}

void GLCommandList::memory_barrier() {
}

void GLCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
}

void GLCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
}

void GLCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
}

void GLCommandList::begin_renderpass(int x0,
                                     int y0,
                                     int x1,
                                     int y1,
                                     uint32_t num_color_attachments,
                                     DeviceAllocation *color_attachments,
                                     bool *color_clear,
                                     std::vector<float> *clear_colors,
                                     DeviceAllocation *depth_attachment,
                                     bool depth_clear) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::end_renderpass() {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::clear_color(float r, float g, float b, float a) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::set_line_width(float width) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::draw_indexed(uint32_t num_indicies,
                                 uint32_t start_vertex,
                                 uint32_t start_index) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::image_transition(DeviceAllocation img,
                                     ImageLayout old_layout,
                                     ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::buffer_to_image(DeviceAllocation dst_img,
                                    DevicePtr src_buf,
                                    ImageLayout img_layout,
                                    const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::image_to_buffer(DevicePtr dst_buf,
                                    DeviceAllocation src_img,
                                    ImageLayout img_layout,
                                    const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

GLStream::~GLStream() {
}

std::unique_ptr<CommandList> GLStream::new_command_list() {
  return std::unique_ptr<CommandList>();
}

void GLStream::submit(CommandList *cmdlist) {
}

void GLStream::submit_synced(CommandList *cmdlist) {
}
void GLStream::command_sync() {
}

GLDevice::~GLDevice() {
}

DeviceAllocation GLDevice::allocate_memory(const AllocParams &params) {
  GLuint buffer;
  glCreateBuffers(1, &buffer);

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = buffer;

  return alloc;
}

void GLDevice::dealloc_memory(DeviceAllocation handle) {
  glDeleteBuffers(1, &handle.alloc_id);
}

std::unique_ptr<Pipeline> GLDevice::create_pipeline(PipelineSourceDesc &src,
                                                    std::string name) {
  return std::make_unique<GLPipeline>(src, name);
}

void *GLDevice::map_range(DevicePtr ptr, uint64_t size) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ptr.alloc_id);
  // Fixme: record the access hint during creation and reflect it here
  return glMapBufferRange(GL_SHADER_STORAGE_BUFFER, ptr.offset, size,
                          GL_READ_WRITE);
}

void *GLDevice::map(DeviceAllocation alloc) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, alloc.alloc_id);
  // Fixme: record the access hint during creation and reflect it here
  return glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
}

void GLDevice::unmap(DevicePtr ptr) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ptr.alloc_id);
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void GLDevice::unmap(DeviceAllocation alloc) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, alloc.alloc_id);
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void GLDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  glBindBuffer(GL_COPY_WRITE_BUFFER, dst.alloc_id);
  glBindBuffer(GL_COPY_READ_BUFFER, src.alloc_id);
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src.offset, dst.offset, size);
}

Stream *GLDevice::get_compute_stream() {
  // Fixme: should we make the GL backend support multi-threading?
  // Or should we ASSERT that we are on the main thread
  return nullptr;
}

std::unique_ptr<Pipeline> GLDevice::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

Stream *GLDevice::get_graphics_stream() {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

std::unique_ptr<Surface> GLDevice::create_surface(const SurfaceConfig &config) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

DeviceAllocation GLDevice::create_image(const ImageParams &params) {
  TI_NOT_IMPLEMENTED;
  return kDeviceNullAllocation;
}

void GLDevice::destroy_image(DeviceAllocation handle) {
  TI_NOT_IMPLEMENTED;
}

void GLDevice::image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void GLDevice::buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void GLDevice::image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

GLSurface::~GLSurface() {
  TI_NOT_IMPLEMENTED;
}

DeviceAllocation GLSurface::get_target_image() {
  TI_NOT_IMPLEMENTED;
  return kDeviceNullAllocation;
}

void GLSurface::present_image() {
  TI_NOT_IMPLEMENTED;
}

std::pair<uint32_t, uint32_t> GLSurface::get_size() {
  TI_NOT_IMPLEMENTED;
  return std::make_pair(0u, 0u);
}

BufferFormat GLSurface::image_format() {
  TI_NOT_IMPLEMENTED;
  return BufferFormat::rgba8;
}

void GLSurface::resize(uint32_t width, uint32_t height) {
  TI_NOT_IMPLEMENTED;
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi