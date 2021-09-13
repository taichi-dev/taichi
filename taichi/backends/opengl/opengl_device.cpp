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
  // FIXME: Implement ranged bind
  TI_NOT_IMPLEMENTED;
}

void GLResourceBinder::rw_buffer(uint32_t set,
                                 uint32_t binding,
                                 DeviceAllocation alloc) {
  TI_ASSERT_INFO(set == 0, "OpenGL only supports set = 0, requested set = {}",
                 set);
  binding_map_[binding] = alloc.alloc_id;
}

void GLResourceBinder::buffer(uint32_t set,
                              uint32_t binding,
                              DevicePtr ptr,
                              size_t size) {
  // FIXME: Implement ranged bind
  TI_NOT_IMPLEMENTED;
}

void GLResourceBinder::buffer(uint32_t set,
                              uint32_t binding,
                              DeviceAllocation alloc) {
  rw_buffer(set, binding, alloc);
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

std::unique_ptr<ResourceBinder::Bindings> GLResourceBinder::materialize() {
  TI_NOT_IMPLEMENTED;
  return nullptr;
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
  return &binder_;
}

GLCommandList::~GLCommandList() {
}

void GLCommandList::bind_pipeline(Pipeline *p) {
  GLPipeline *pipeline = static_cast<GLPipeline *>(p);
  auto cmd = std::make_unique<CmdBindPipeline>();
  cmd->program = pipeline->get_program();
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::bind_resources(ResourceBinder *_binder) {
  GLResourceBinder *binder = static_cast<GLResourceBinder *>(_binder);
  for (auto &[binding, buffer] : binder->binding_map()) {
    auto cmd = std::make_unique<CmdBindBufferToIndex>();
    cmd->buffer = buffer;
    cmd->index = binding;
    recorded_commands_.push_back(std::move(cmd));
  }
}

void GLCommandList::bind_resources(ResourceBinder *binder,
                                   ResourceBinder::Bindings *bindings) {
  TI_NOT_IMPLEMENTED;
}

template <typename T>
std::initializer_list<T> make_init_list(std::initializer_list<T> &&l) {
  return l;
}

void GLCommandList::buffer_barrier(DevicePtr ptr, size_t size) {
  recorded_commands_.push_back(std::make_unique<CmdBufferBarrier>());
}

void GLCommandList::buffer_barrier(DeviceAllocation alloc) {
  recorded_commands_.push_back(std::make_unique<CmdBufferBarrier>());
}

void GLCommandList::memory_barrier() {
  recorded_commands_.push_back(std::make_unique<CmdBufferBarrier>());
}

void GLCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  auto cmd = std::make_unique<CmdBufferCopy>();
  cmd->src = src.alloc_id;
  cmd->dst = dst.alloc_id;
  cmd->src_offset = src.offset;
  cmd->dst_offset = dst.offset;
  cmd->size = size;
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  auto cmd = std::make_unique<CmdBufferFill>();
  cmd->buffer = ptr.alloc_id;
  cmd->size = size;
  cmd->data = data;
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  auto cmd = std::make_unique<CmdDispatch>();
  cmd->x = x;
  cmd->y = y;
  cmd->z = z;
  recorded_commands_.push_back(std::move(cmd));
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

void GLCommandList::run_commands() {
  for (auto &cmd : recorded_commands_) {
    cmd->execute();
  }
}

GLStream::~GLStream() {
}

std::unique_ptr<CommandList> GLStream::new_command_list() {
  return std::make_unique<GLCommandList>();
}

void GLStream::submit(CommandList *_cmdlist) {
  GLCommandList *cmdlist = static_cast<GLCommandList *>(_cmdlist);
  cmdlist->run_commands();
}

void GLStream::submit_synced(CommandList *cmdlist) {
  submit(cmdlist);
  glFinish();
}
void GLStream::command_sync() {
  glFinish();
}

GLDevice::~GLDevice() {
}

DeviceAllocation GLDevice::allocate_memory(const AllocParams &params) {
  GLuint buffer;
  glCreateBuffers(1, &buffer);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, params.size, nullptr, GL_DYNAMIC_READ);

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
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src.offset,
                      dst.offset, size);
  glFinish();
}

Stream *GLDevice::get_compute_stream() {
  // Fixme: should we make the GL backend support multi-threading?
  // Or should we ASSERT that we are on the main thread
  return &stream_;
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

void GLCommandList::CmdBindPipeline::execute() {
  glUseProgram(program);
}

void GLCommandList::CmdBindBufferToIndex::execute() {
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer);
}

void GLCommandList::CmdBufferBarrier::execute() {
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GLCommandList::CmdBufferCopy::execute() {
  glBindBuffer(GL_COPY_READ_BUFFER, src);
  glBindBuffer(GL_COPY_WRITE_BUFFER, dst);
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src_offset,
                      dst_offset, size);
}

void GLCommandList::CmdBufferFill::execute() {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
  glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_UNSIGNED_INT, 0, size,
                       GL_UNSIGNED_INT, GL_RED_INTEGER, &data);
}

void GLCommandList::CmdDispatch::execute() {
  glDispatchCompute(x, y, z);
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi