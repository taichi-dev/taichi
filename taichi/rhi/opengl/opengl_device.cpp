#include "opengl_device.h"
#include "opengl_api.h"

#include "spirv_glsl.hpp"

namespace taichi::lang {
namespace opengl {

namespace {
const std::unordered_map<BufferFormat, GLuint> format_to_gl_internal_format = {
    {BufferFormat::r8, GL_R8},
    {BufferFormat::rg8, GL_RG8},
    {BufferFormat::rgba8, GL_RGBA8},
    {BufferFormat::rgba8srgb, GL_SRGB8_ALPHA8},
    {BufferFormat::bgra8, GL_BGRA8_EXT},
    {BufferFormat::bgra8srgb, GL_INVALID_ENUM},
    {BufferFormat::r8u, GL_R8UI},
    {BufferFormat::rg8u, GL_RG8UI},
    {BufferFormat::rgba8u, GL_RGBA8UI},
    {BufferFormat::r8i, GL_R8I},
    {BufferFormat::rg8i, GL_RG8I},
    {BufferFormat::rgba8i, GL_RGBA8I},
    {BufferFormat::r16, GL_R16},
    {BufferFormat::rg16, GL_RG16},
    {BufferFormat::rgb16, GL_RGB16},
    {BufferFormat::rgba16, GL_RGBA16},
    {BufferFormat::r16u, GL_R16UI},
    {BufferFormat::rg16u, GL_RG16UI},
    {BufferFormat::rgb16u, GL_RGB16UI},
    {BufferFormat::rgba16u, GL_RGBA16UI},
    {BufferFormat::r16i, GL_R16I},
    {BufferFormat::rg16i, GL_RG16I},
    {BufferFormat::rgb16i, GL_RGB16I},
    {BufferFormat::rgba16i, GL_RGBA16I},
    {BufferFormat::r16f, GL_R16F},
    {BufferFormat::rg16f, GL_RG16F},
    {BufferFormat::rgb16f, GL_RGB16F},
    {BufferFormat::rgba16f, GL_RGBA16F},
    {BufferFormat::r32u, GL_R32UI},
    {BufferFormat::rg32u, GL_RG32UI},
    {BufferFormat::rgb32u, GL_RGB32UI},
    {BufferFormat::rgba32u, GL_RGBA32UI},
    {BufferFormat::r32i, GL_R32I},
    {BufferFormat::rg32i, GL_RG32I},
    {BufferFormat::rgb32i, GL_RGB32I},
    {BufferFormat::rgba32i, GL_RGBA32I},
    {BufferFormat::r32f, GL_R32F},
    {BufferFormat::rg32f, GL_RG32F},
    {BufferFormat::rgb32f, GL_RGB32F},
    {BufferFormat::rgba32f, GL_RGBA32F},
    {BufferFormat::depth16, GL_INVALID_ENUM},
    {BufferFormat::depth24stencil8, GL_DEPTH24_STENCIL8},
    {BufferFormat::depth32f, GL_DEPTH32F_STENCIL8}};

const std::unordered_map<GLuint, GLuint> gl_internal_format_to_type = {
    {GL_R8, GL_UNSIGNED_BYTE},
    {GL_R8_SNORM, GL_BYTE},
    {GL_R8UI, GL_UNSIGNED_BYTE},
    {GL_R8I, GL_BYTE},
    {GL_R16, GL_UNSIGNED_SHORT},
    {GL_R16_SNORM, GL_SHORT},
    {GL_R16F, GL_HALF_FLOAT},
    {GL_R16UI, GL_UNSIGNED_SHORT},
    {GL_R16I, GL_SHORT},
    {GL_R32UI, GL_UNSIGNED_INT},
    {GL_R32I, GL_INT},
    {GL_R32F, GL_FLOAT},
    {GL_RG8, GL_UNSIGNED_BYTE},
    {GL_RG8_SNORM, GL_BYTE},
    {GL_RG8UI, GL_UNSIGNED_BYTE},
    {GL_RG8I, GL_BYTE},
    {GL_RG16, GL_UNSIGNED_SHORT},
    {GL_RG16_SNORM, GL_SHORT},
    {GL_RG16F, GL_HALF_FLOAT},
    {GL_RG16UI, GL_UNSIGNED_SHORT},
    {GL_RG16I, GL_SHORT},
    {GL_RG32UI, GL_UNSIGNED_INT},
    {GL_RG32I, GL_INT},
    {GL_RG32F, GL_FLOAT},
    {GL_RGB8, GL_UNSIGNED_BYTE},
    {GL_RGB8_SNORM, GL_BYTE},
    {GL_RGB8UI, GL_UNSIGNED_BYTE},
    {GL_RGB8I, GL_BYTE},
    {GL_RGB16, GL_UNSIGNED_SHORT},
    {GL_RGB16_SNORM, GL_SHORT},
    {GL_RGB16F, GL_HALF_FLOAT},
    {GL_RGB16UI, GL_UNSIGNED_SHORT},
    {GL_RGB16I, GL_SHORT},
    {GL_RGB32UI, GL_UNSIGNED_INT},
    {GL_RGB32I, GL_INT},
    {GL_RGB32F, GL_FLOAT},
    {GL_RGBA8, GL_UNSIGNED_BYTE},
    {GL_SRGB8_ALPHA8, GL_UNSIGNED_BYTE},
    {GL_RGBA8_SNORM, GL_BYTE},
    {GL_RGBA8UI, GL_UNSIGNED_BYTE},
    {GL_RGBA8I, GL_BYTE},
    {GL_RGBA16, GL_UNSIGNED_SHORT},
    {GL_RGBA16_SNORM, GL_SHORT},
    {GL_RGBA16F, GL_HALF_FLOAT},
    {GL_RGBA16UI, GL_UNSIGNED_SHORT},
    {GL_RGBA16I, GL_SHORT},
    {GL_RGBA32UI, GL_UNSIGNED_INT},
    {GL_RGBA32I, GL_INT},
    {GL_RGBA32F, GL_FLOAT}};

const std::unordered_map<GLuint, GLuint> gl_internal_format_to_format = {
    {GL_R8, GL_RED},
    {GL_R8_SNORM, GL_RED},
    {GL_R8UI, GL_RED},
    {GL_R8I, GL_RED},
    {GL_R16, GL_RED},
    {GL_R16_SNORM, GL_RED},
    {GL_R16F, GL_RED},
    {GL_R16UI, GL_RED},
    {GL_R16I, GL_RED},
    {GL_R32UI, GL_RED},
    {GL_R32I, GL_RED},
    {GL_R32F, GL_RED},
    {GL_RG8, GL_RG},
    {GL_RG8_SNORM, GL_RG},
    {GL_RG8UI, GL_RG},
    {GL_RG8I, GL_RG},
    {GL_RG16, GL_RG},
    {GL_RG16_SNORM, GL_RG},
    {GL_RG16F, GL_RG},
    {GL_RG16UI, GL_RG},
    {GL_RG16I, GL_RG},
    {GL_RG32UI, GL_RG},
    {GL_RG32I, GL_RG},
    {GL_RG32F, GL_RG},
    {GL_RGB8, GL_RGB},
    {GL_RGB8_SNORM, GL_RGB},
    {GL_RGB8UI, GL_RGB},
    {GL_RGB8I, GL_RGB},
    {GL_RGB16, GL_RGB},
    {GL_RGB16_SNORM, GL_RGB},
    {GL_RGB16F, GL_RGB},
    {GL_RGB16UI, GL_RGB},
    {GL_RGB16I, GL_RGB},
    {GL_RGB32UI, GL_RGB},
    {GL_RGB32I, GL_RGB},
    {GL_RGB32F, GL_RGB},
    {GL_RGBA8, GL_RGBA},
    {GL_SRGB8_ALPHA8, GL_RGBA},
    {GL_RGBA8_SNORM, GL_RGBA},
    {GL_RGBA8UI, GL_RGBA},
    {GL_RGBA8I, GL_RGBA},
    {GL_RGBA16, GL_RGBA},
    {GL_RGBA16_SNORM, GL_RGBA},
    {GL_RGBA16F, GL_RGBA},
    {GL_RGBA16UI, GL_RGBA},
    {GL_RGBA16I, GL_RGBA},
    {GL_RGBA32UI, GL_RGBA},
    {GL_RGBA32I, GL_RGBA},
    {GL_RGBA32F, GL_RGBA}};
}  // namespace

std::string get_opengl_error_string(GLenum err) {
  switch (err) {
#define PER_GL_ERR(x) \
  case x:             \
    return #x;
    PER_GL_ERR(GL_NO_ERROR)
    PER_GL_ERR(GL_INVALID_ENUM)
    PER_GL_ERR(GL_INVALID_VALUE)
    PER_GL_ERR(GL_INVALID_OPERATION)
    PER_GL_ERR(GL_INVALID_FRAMEBUFFER_OPERATION)
    PER_GL_ERR(GL_OUT_OF_MEMORY)
    PER_GL_ERR(GL_STACK_UNDERFLOW)
    PER_GL_ERR(GL_STACK_OVERFLOW)
    default:
      return fmt::format("GL_ERROR={}", err);
  }
#undef PER_GL_ERR
}

GLResourceSet::~GLResourceSet() {
}

GLResourceSet &GLResourceSet::rw_buffer(uint32_t binding,
                                        DevicePtr ptr,
                                        size_t size) {
  ssbo_binding_map_[binding] = {GLuint(ptr.alloc_id), ptr.offset, size};
  return *this;
}

GLResourceSet &GLResourceSet::rw_buffer(uint32_t binding,
                                        DeviceAllocation alloc) {
  return rw_buffer(binding, alloc.get_ptr(0), -1);
}

GLResourceSet &GLResourceSet::buffer(uint32_t binding,
                                     DevicePtr ptr,
                                     size_t size) {
  ubo_binding_map_[binding] = {GLuint(ptr.alloc_id), ptr.offset, size};
  return *this;
}

GLResourceSet &GLResourceSet::buffer(uint32_t binding, DeviceAllocation alloc) {
  return buffer(binding, alloc.get_ptr(0), -1);
}

GLResourceSet &GLResourceSet::image(uint32_t binding,
                                    DeviceAllocation alloc,
                                    ImageSamplerConfig sampler_config) {
  texture_binding_map_[binding] = alloc.alloc_id;
  return *this;
}

GLResourceSet &GLResourceSet::rw_image(uint32_t binding,
                                       DeviceAllocation alloc,
                                       int lod) {
  rw_image_binding_map_[binding] = alloc.alloc_id;
  return *this;
}

GLPipeline::GLPipeline(const PipelineSourceDesc &desc,
                       const std::string &name) {
  GLuint shader_id;
  shader_id = glCreateShader(GL_COMPUTE_SHADER);
  check_opengl_error("glCreateShader");

  if (desc.type == PipelineSourceType::glsl_src) {
    const GLchar *source_cstr = (const GLchar *)desc.data;
    int length = desc.size;
    glShaderSource(shader_id, 1, &source_cstr, &length);
    check_opengl_error("glShaderSource");
  } else if (desc.type == PipelineSourceType::spirv_binary) {
    spirv_cross::CompilerGLSL glsl((uint32_t *)desc.data,
                                   desc.size / sizeof(uint32_t));
    spirv_cross::CompilerGLSL::Options options;
    if (is_gles()) {
      options.es = true;
      options.version = 310;
    } else {
      options.enable_420pack_extension = true;
    }
    options.vulkan_semantics = false;
    glsl.set_common_options(options);
    if (GLAD_GL_NV_gpu_shader5) {
      glsl.require_extension("GL_NV_gpu_shader5");
      glsl.add_header_line(
          "uint16_t float16BitsToUint16(float16_t v) { return "
          "uint16_t(packFloat2x16(f16vec2(v, 0))); }");
      glsl.add_header_line(
          "float16_t uint16BitsToFloat16(uint16_t v) { return "
          "unpackFloat2x16(uint(v)).x; }");
    }
    glsl.add_header_line("");
    std::string source = glsl.compile();
    TI_TRACE("GLSL source: \n{}", source);

    const char *src = source.data();
    GLint length = GLint(source.length());
    glShaderSource(shader_id, 1, &src, &length);
    check_opengl_error("glShaderSource");
  } else {
    TI_ERROR("Pipeline source type not supported");
  }

  glCompileShader(shader_id);
  check_opengl_error("glCompileShader");
  int status = GL_TRUE;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
  check_opengl_error("glGetShaderiv");
  if (status != GL_TRUE) {
    GLsizei logLength;
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &logLength);
    check_opengl_error("glGetShaderiv");

    auto log = std::vector<GLchar>(logLength + 1);
    glGetShaderInfoLog(shader_id, logLength, &logLength, log.data());
    check_opengl_error("glGetShaderInfoLog");

    log[logLength] = 0;
    TI_ERROR("[glsl] error while compiling shader:\n{}", log.data());
  }
  check_opengl_error("shader compile");

  program_id_ = glCreateProgram();
  check_opengl_error("glCreateProgram");
  glAttachShader(program_id_, shader_id);
  check_opengl_error("glAttachShader");
  glLinkProgram(program_id_);
  check_opengl_error("glLinkProgram");
  glGetProgramiv(program_id_, GL_LINK_STATUS, &status);
  check_opengl_error("glGetProgramiv");
  if (status != GL_TRUE) {
    GLsizei logLength;
    glGetProgramiv(program_id_, GL_INFO_LOG_LENGTH, &logLength);
    check_opengl_error("glGetProgramiv");
    auto log = std::vector<GLchar>(logLength + 1);
    glGetProgramInfoLog(program_id_, logLength, &logLength, log.data());
    check_opengl_error("glGetProgramInfoLog");
    log[logLength] = 0;
    TI_ERROR("[glsl] error while linking program:\n{}", log.data());
  }

  glDeleteShader(shader_id);
  check_opengl_error("glDeleteShader");
}

GLPipeline::~GLPipeline() {
  glDeleteProgram(program_id_);
  check_opengl_error("glDeleteShader");
}

GLCommandList::~GLCommandList() {
}

void GLCommandList::bind_pipeline(Pipeline *p) noexcept {
  GLPipeline *pipeline = static_cast<GLPipeline *>(p);
  auto cmd = std::make_unique<CmdBindPipeline>();
  cmd->program = pipeline->get_program();
  recorded_commands_.push_back(std::move(cmd));
}

RhiResult GLCommandList::bind_shader_resources(ShaderResourceSet *res,
                                               int set_index) noexcept {
  GLResourceSet *set = static_cast<GLResourceSet *>(res);
  auto cmd = std::make_unique<CmdBindResources>();
  for (auto &[binding, buffer] : set->ssbo_binding_map()) {
    auto &bind = cmd->buffers.emplace_back();
    bind.buffer = buffer.buffer;
    bind.offset = buffer.offset;
    bind.size = buffer.size;
    bind.index = binding;
  }
  for (auto &[binding, buffer] : set->ubo_binding_map()) {
    auto &bind = cmd->buffers.emplace_back();
    bind.buffer = buffer.buffer;
    bind.offset = buffer.offset;
    bind.size = buffer.size;
    bind.index = binding;
    bind.target = GL_UNIFORM_BUFFER;
  }
  for (auto &[binding, texture] : set->texture_binding_map()) {
    auto &bind = cmd->textures.emplace_back();
    bind.texture = texture;
    bind.index = binding;
    bind.target = device_->get_gl_image(texture).target;
  }
  for (auto &[binding, texture] : set->rw_image_binding_map()) {
    auto &bind = cmd->textures.emplace_back();
    bind.texture = texture;
    bind.index = binding;
    bind.format = device_->get_gl_image(texture).format;
    bind.is_storage = true;
  }

  recorded_commands_.push_back(std::move(cmd));
  return RhiResult::success;
}

RhiResult GLCommandList::bind_raster_resources(RasterResources *res) noexcept {
  TI_NOT_IMPLEMENTED;
}

void GLCommandList::buffer_barrier(DevicePtr ptr, size_t size) noexcept {
  recorded_commands_.push_back(std::make_unique<CmdBufferBarrier>());
}

void GLCommandList::buffer_barrier(DeviceAllocation alloc) noexcept {
  recorded_commands_.push_back(std::make_unique<CmdBufferBarrier>());
}

void GLCommandList::memory_barrier() noexcept {
  recorded_commands_.push_back(std::make_unique<CmdBufferBarrier>());
}

void GLCommandList::buffer_copy(DevicePtr dst,
                                DevicePtr src,
                                size_t size) noexcept {
  auto cmd = std::make_unique<CmdBufferCopy>();
  cmd->src = src.alloc_id;
  cmd->dst = dst.alloc_id;
  cmd->src_offset = src.offset;
  cmd->dst_offset = dst.offset;
  cmd->size = size;
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::buffer_fill(DevicePtr ptr,
                                size_t size,
                                uint32_t data) noexcept {
  auto cmd = std::make_unique<CmdBufferFill>();
  cmd->buffer = ptr.alloc_id;
  cmd->offset = ptr.offset;
  cmd->size = size;
  cmd->data = data;
  recorded_commands_.push_back(std::move(cmd));
}

RhiResult GLCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) noexcept {
  auto cmd = std::make_unique<CmdDispatch>();
  cmd->x = x;
  cmd->y = y;
  cmd->z = z;
  recorded_commands_.push_back(std::move(cmd));
  return RhiResult::success;
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
  auto cmd = std::make_unique<CmdImageTransition>();
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::buffer_to_image(DeviceAllocation dst_img,
                                    DevicePtr src_buf,
                                    ImageLayout img_layout,
                                    const BufferImageCopyParams &params) {
  auto cmd = std::make_unique<CmdBufferToImage>();
  cmd->params = params;
  cmd->image = dst_img.alloc_id;
  cmd->buffer = src_buf.alloc_id;
  cmd->offset = src_buf.offset;
  cmd->device = device_;
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::image_to_buffer(DevicePtr dst_buf,
                                    DeviceAllocation src_img,
                                    ImageLayout img_layout,
                                    const BufferImageCopyParams &params) {
  auto cmd = std::make_unique<CmdImageToBuffer>();
  cmd->params = params;
  cmd->image = src_img.alloc_id;
  cmd->buffer = dst_buf.alloc_id;
  cmd->offset = dst_buf.offset;
  cmd->device = device_;
  recorded_commands_.push_back(std::move(cmd));
}

void GLCommandList::run_commands() {
  for (auto &cmd : recorded_commands_) {
    cmd->execute();
  }
}

GLStream::~GLStream() {
}

RhiResult GLStream::new_command_list(CommandList **out_cmdlist) noexcept {
  *out_cmdlist = new GLCommandList(device_);
  return RhiResult::success;
}

StreamSemaphore GLStream::submit(
    CommandList *_cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  GLCommandList *cmdlist = static_cast<GLCommandList *>(_cmdlist);
  cmdlist->run_commands();

  // OpenGL is fully serial
  return nullptr;
}

StreamSemaphore GLStream::submit_synced(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  submit(cmdlist);
  glFinish();
  check_opengl_error("glFinish");

  // OpenGL is fully serial
  return nullptr;
}
void GLStream::command_sync() {
  glFinish();
  check_opengl_error("glFinish");
}

GLDevice::GLDevice() : stream_(this) {
  initialize_opengl(false, true);
  DeviceCapabilityConfig caps{};
  if (!is_gles()) {
    // 64bit isn't supported in ES profile
    caps.set(DeviceCapability::spirv_has_int64, true);
    caps.set(DeviceCapability::spirv_has_float64, true);
  }

  if (GLAD_GL_NV_gpu_shader5) {
    caps.set(DeviceCapability::spirv_has_int16, true);
    caps.set(DeviceCapability::spirv_has_float16, true);
  }

  if (GLAD_GL_AMD_gpu_shader_int16) {
    caps.set(DeviceCapability::spirv_has_int16, true);
  }

  if (GLAD_GL_AMD_gpu_shader_half_float) {
    caps.set(DeviceCapability::spirv_has_float16, true);
  }

  caps.set(DeviceCapability::spirv_version, 0x10300);
  set_caps(std::move(caps));
}

GLDevice::~GLDevice() {
}

RhiResult GLDevice::allocate_memory(const AllocParams &params,
                                    DeviceAllocation *out_devalloc) {
  GLenum target_hint = GL_SHADER_STORAGE_BUFFER;
  if (params.usage && AllocUsage::Storage) {
    target_hint = GL_SHADER_STORAGE_BUFFER;
  } else if (params.usage && AllocUsage::Uniform) {
    target_hint = GL_UNIFORM_BUFFER;
  }
  GLuint buffer;
  glGenBuffers(1, &buffer);
  check_opengl_error("glGenBuffers");
  glBindBuffer(target_hint, buffer);
  check_opengl_error("glBindBuffer");

  glBufferData(target_hint, params.size, nullptr,
               params.host_read ? GL_STATIC_COPY : GL_DYNAMIC_READ);
  GLuint alloc_res = glGetError();

  if (alloc_res == GL_OUT_OF_MEMORY) {
    return RhiResult::out_of_memory;
  }
  check_opengl_error("glBufferData");

  *out_devalloc = DeviceAllocation{};
  out_devalloc->device = this;
  out_devalloc->alloc_id = buffer;

  if (params.host_read && params.host_write) {
    buffer_to_access_[buffer] = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
  } else if (params.host_read) {
    buffer_to_access_[buffer] = GL_MAP_READ_BIT;
  } else if (params.host_write) {
    buffer_to_access_[buffer] = GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT;
  }

  return RhiResult::success;
}

void GLDevice::dealloc_memory(DeviceAllocation handle) {
  GLuint buffer = GLuint(handle.alloc_id);
  glDeleteBuffers(1, &buffer);
  check_opengl_error("glDeleteBuffers");
}

GLint GLDevice::get_devalloc_size(DeviceAllocation handle) {
  glBindBuffer(GL_ARRAY_BUFFER, handle.alloc_id);
  check_opengl_error("glBindBuffer");
  GLint size = 0;
  glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
  check_opengl_error("glGetBufferParameteriv");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  check_opengl_error("glBindBuffer");
  return size;
}

RhiResult GLDevice::upload_data(DevicePtr *device_ptr,
                                const void **data,
                                size_t *size,
                                int num_alloc) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }

    glBindBuffer(GL_ARRAY_BUFFER, device_ptr[i].alloc_id);
    check_opengl_error("glBindBuffer");
    glBufferSubData(GL_ARRAY_BUFFER, device_ptr[i].offset, GLsizeiptr(size[i]),
                    data[i]);
    check_opengl_error("glBufferSubData");
  }

  return RhiResult::success;
}

RhiResult GLDevice::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }

    glBindBuffer(GL_ARRAY_BUFFER, device_ptr[i].alloc_id);
    check_opengl_error("glBindBuffer");
    void *ptr = glMapBufferRange(GL_ARRAY_BUFFER, device_ptr[i].offset,
                                 GLsizeiptr(size[i]), GL_MAP_READ_BIT);
    check_opengl_error("glMapBufferRange");
    memcpy(data[i], ptr, size[i]);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    check_opengl_error("glUnmapBuffer");
  }

  return RhiResult::success;
}

RhiResult GLDevice::create_pipeline(Pipeline **out_pipeline,
                                    const PipelineSourceDesc &src,
                                    std::string name,
                                    PipelineCache *cache) noexcept {
  try {
    *out_pipeline = new GLPipeline(src, name);
  } catch (std::bad_alloc &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::out_of_memory;
  } catch (std::invalid_argument &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::invalid_usage;
  } catch (std::runtime_error &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::error;
  }
  return RhiResult::success;
}

RhiResult GLDevice::map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) {
  TI_ASSERT_INFO(
      buffer_to_access_.find(ptr.alloc_id) != buffer_to_access_.end(),
      "Buffer not created with host_read or write");
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ptr.alloc_id);
  check_opengl_error("glBindBuffer");
  *mapped_ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, ptr.offset, size,
                                 buffer_to_access_.at(ptr.alloc_id));
  check_opengl_error("glMapBufferRange");
  return RhiResult::success;
}

RhiResult GLDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, alloc.alloc_id);
  check_opengl_error("glBindBuffer");

  int size = 0;
  glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &size);
  check_opengl_error("glGetBufferParameteriv");

  return map_range(alloc.get_ptr(0), size, mapped_ptr);
}

void GLDevice::unmap(DevicePtr ptr) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ptr.alloc_id);
  check_opengl_error("glBindBuffer");
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  check_opengl_error("glUnmapBuffer");
}

void GLDevice::unmap(DeviceAllocation alloc) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, alloc.alloc_id);
  check_opengl_error("glBindBuffer");
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  check_opengl_error("glUnmapBuffer");
}

void GLDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_ASSERT(dst.device == src.device);
  glBindBuffer(GL_COPY_WRITE_BUFFER, dst.alloc_id);
  check_opengl_error("glBindBuffer");
  glBindBuffer(GL_COPY_READ_BUFFER, src.alloc_id);
  check_opengl_error("glBindBuffer");
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src.offset,
                      dst.offset, size);
  check_opengl_error("glCopyBufferSubData");
  glFinish();
  check_opengl_error("glFinish");
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

void GLDevice::wait_idle() {
  glFinish();
  check_opengl_error("glFinish");
}

std::unique_ptr<Surface> GLDevice::create_surface(const SurfaceConfig &config) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

DeviceAllocation GLDevice::create_image(const ImageParams &params) {
  GLuint tex;
  glGenTextures(1, &tex);
  check_opengl_error("glGenTextures");

  auto gl_texture_dims = GL_TEXTURE_2D;
  if (params.dimension == ImageDimension::d1D) {
    gl_texture_dims = GL_TEXTURE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    gl_texture_dims = GL_TEXTURE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    gl_texture_dims = GL_TEXTURE_3D;
  }

  auto format = format_to_gl_internal_format.at(params.format);

  glBindTexture(gl_texture_dims, tex);
  check_opengl_error("glBindTexture");
  if (params.dimension == ImageDimension::d1D) {
    glTexStorage1D(gl_texture_dims, 1, format, params.x);
    check_opengl_error("glTexStorage1D");
  } else if (params.dimension == ImageDimension::d2D) {
    glTexStorage2D(gl_texture_dims, 1, format, params.x, params.y);
    check_opengl_error("glTexStorage2D");
  } else {
    glTexStorage3D(gl_texture_dims, 1, format, params.x, params.y, params.z);
    check_opengl_error("glTexStorage3D");
  }

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = tex;

  GLImageAllocation gl_image{};
  gl_image.target = gl_texture_dims;
  gl_image.levels = 1;
  gl_image.format = format;
  gl_image.width = params.x;
  gl_image.height = params.y;
  gl_image.depth = params.z;
  gl_image.external = false;

  image_allocs_[tex] = std::move(gl_image);
  return alloc;
}

void GLDevice::destroy_image(DeviceAllocation handle) {
  GLuint texture = GLuint(handle.alloc_id);
  auto it = image_allocs_.find(texture);
  if (it != image_allocs_.end()) {
    if (!it->second.external) {
      glDeleteTextures(1, &texture);
      check_opengl_error("glDeleteTextures");
    }
    image_allocs_.erase(it);
  }
}

DeviceAllocation GLDevice::import_image(GLuint texture,
                                        GLImageAllocation &&gl_image) {
  image_allocs_[texture] = std::move(gl_image);

  DeviceAllocation out{};
  out.device = this;
  out.alloc_id = (DeviceAllocationId)texture;
  return out;
}

void GLDevice::image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout) {
  glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT |
                  GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                  GL_FRAMEBUFFER_BARRIER_BIT);
  check_opengl_error("glMemoryBarrier");
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
}

StreamSemaphore GLSurface::acquire_next_image() {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

DeviceAllocation GLSurface::get_target_image() {
  TI_NOT_IMPLEMENTED;
  return kDeviceNullAllocation;
}

void GLSurface::present_image(
    const std::vector<StreamSemaphore> &wait_semaphores) {
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
  check_opengl_error("glUseProgram");
}

void GLCommandList::CmdBindResources::execute() {
  for (auto &bind : buffers) {
    if (bind.size == -1) {
      glBindBufferBase(bind.target, bind.index, bind.buffer);
      check_opengl_error("glBindBufferBase");
    } else {
      glBindBufferRange(bind.target, bind.index, bind.buffer,
                        GLintptr(bind.offset), GLsizeiptr(bind.size));
      check_opengl_error("glBindBufferRange");
    }
  }
  for (auto &bind : textures) {
    if (bind.is_storage) {
      glBindImageTexture(bind.index, bind.texture, /*level=*/0,
                         /*layered=*/GL_FALSE, /*layer=*/0, GL_READ_WRITE,
                         bind.format);
      check_opengl_error("glBindImageTexture");
    } else {
      glActiveTexture(GL_TEXTURE0 + bind.index);
      check_opengl_error("glActiveTexture");
      glBindTexture(bind.target, bind.texture);
      check_opengl_error("glBindTexture");
    }
  }
}

void GLCommandList::CmdBufferBarrier::execute() {
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT |
                  GL_UNIFORM_BARRIER_BIT);
  check_opengl_error("glMemoryBarrier");
}

void GLCommandList::CmdBufferCopy::execute() {
  glBindBuffer(GL_COPY_READ_BUFFER, src);
  check_opengl_error("glBindBuffer");
  int buf_size = 0;
  glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &buf_size);
  check_opengl_error("glGetBufferParameteriv");
  glBindBuffer(GL_COPY_WRITE_BUFFER, dst);
  check_opengl_error("glBindBuffer");
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, src_offset,
                      dst_offset,
                      (kBufferSizeEntireSize == size) ? buf_size : size);
  check_opengl_error("glCopyBufferSubData");
}

void GLCommandList::CmdBufferFill::execute() {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
  check_opengl_error("glBindBuffer");
  int buf_size = 0;
  glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &buf_size);
  check_opengl_error("glGetBufferParameteriv");
  if (is_gles()) {
    TI_ASSERT_INFO(offset == 0 && data == 0 && size >= buf_size,
                   "GLES only supports full clear");
    glBufferData(GL_SHADER_STORAGE_BUFFER, buf_size, nullptr, GL_DYNAMIC_READ);
    check_opengl_error("glBufferData");
  } else {
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32F, offset,
                         (kBufferSizeEntireSize == size) ? buf_size : size,
                         GL_RED, GL_FLOAT, &data);
    check_opengl_error("glClearBufferSubData");
  }
}

void GLCommandList::CmdDispatch::execute() {
  glDispatchCompute(x, y, z);
  check_opengl_error("glDispatchCompute");
}

void GLCommandList::CmdImageTransition::execute() {
  glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT |
                  GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                  GL_FRAMEBUFFER_BARRIER_BIT);
  check_opengl_error("glMemoryBarrier");
}

void GLCommandList::CmdBufferToImage::execute() {
  const GLImageAllocation &gl_image = device->get_gl_image(image);
  GLuint image_dims = gl_image.target;
  GLuint image_internal_format = gl_image.format;
  GLuint image_format = gl_internal_format_to_format.at(image_internal_format);
  GLuint gl_type = gl_internal_format_to_type.at(image_internal_format);

  glBindTexture(image_dims, image);
  check_opengl_error("glBindTexture");
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
  check_opengl_error("glBindBuffer");
  if (image_dims == GL_TEXTURE_1D) {
    glTexSubImage1D(image_dims, /*level=*/0, params.image_offset.x,
                    params.image_extent.y, image_format, gl_type,
                    (void *)offset);
  } else if (image_dims == GL_TEXTURE_2D) {
    glTexSubImage2D(image_dims, /*level=*/0, /*xoffset=*/params.image_offset.x,
                    /*yoffset=*/params.image_offset.y,
                    /*width=*/params.image_extent.x,
                    /*height=*/params.image_extent.y, image_format, gl_type,
                    (void *)offset);
  } else {
    glTexSubImage3D(image_dims, /*level=*/0, /*xoffset=*/params.image_offset.x,
                    params.image_offset.y, params.image_offset.z,
                    params.image_extent.x, params.image_extent.y,
                    params.image_extent.z, image_format, gl_type,
                    (void *)offset);
  }
  check_opengl_error("glTexSubImage");
  glBindTexture(image_dims, /*target=*/0);
  check_opengl_error("glBindTexture");
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, /*target=*/0);
  check_opengl_error("glBindBuffer");
}

void GLCommandList::CmdImageToBuffer::execute() {
  const GLImageAllocation &gl_image = device->get_gl_image(image);
  auto image_dims = gl_image.target;
  auto image_format = gl_image.format;
  auto gl_type = gl_internal_format_to_type.at(image_format);
  auto unsized_format = gl_internal_format_to_format.at(image_format);

  glBindTexture(image_dims, image);
  check_opengl_error("glBindTexture");
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
  check_opengl_error("glBindBuffer");
  TI_ASSERT_INFO(params.image_offset.x == 0 && params.image_offset.y == 0 &&
                     params.image_offset.z == 0,
                 "OpenGL can only copy full images to buffer");
  glGetTexImage(/*level=*/0, image_format, unsized_format, gl_type,
                (void *)offset);
  check_opengl_error("glGetTexImage");
  glBindTexture(image_dims, /*target=*/0);
  check_opengl_error("glBindTexture");
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, /*target=*/0);
  check_opengl_error("glBindBuffer");
}

}  // namespace opengl
}  // namespace taichi::lang
