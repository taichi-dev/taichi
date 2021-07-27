//#define _GLSL_DEBUG 1
#include "opengl_api.h"

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/util/environ_config.h"
#include "taichi/backends/opengl/shaders/runtime.h"
#include "taichi/backends/opengl/shaders/listman.h"
#include "taichi/ir/transforms.h"

#ifdef TI_WITH_OPENGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#endif

#include <list>

TLANG_NAMESPACE_BEGIN
namespace opengl {

#define PER_OPENGL_EXTENSION(x) bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

// will later be initialized in initialize_opengl, here we use the minimum
// value according to OpenGL spec in case glGetIntegerv didn't work properly
int opengl_max_block_dim = 1024;
int opengl_max_grid_dim = 1024;

#ifdef TI_WITH_OPENGL

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
}

void check_opengl_error(const std::string &msg = "OpenGL") {
  auto err = glGetError();
  if (err != GL_NO_ERROR) {
    auto estr = get_opengl_error_string(err);
    TI_ERROR("{}: {}", msg, estr);
  }
}

static std::string add_line_markers(std::string x) {
  std::string marker;
  size_t pos = 0, npos;
  int line = 0;
  while (1) {
    npos = x.find_first_of('\n', pos);
    marker = fmt::format("{:3d} ", ++line);
    if (npos == std::string::npos)
      break;
    x.insert(pos, marker);
    pos = npos + 1 + marker.size();
  }
  return x;
}

struct GLShader {
  GLuint id_;

  GLShader(GLuint type = GL_COMPUTE_SHADER) {
    id_ = glCreateShader(type);
  }

  GLShader(const std::string &source, GLuint type = GL_COMPUTE_SHADER)
      : GLShader(type) {
    this->compile(source);
  }

  ~GLShader() {
    glDeleteShader(id_);
  }

  void compile(const std::string &source) const {
    const GLchar *source_cstr = source.c_str();
    glShaderSource(id_, 1, &source_cstr, nullptr);

    TI_TRACE("glCompileShader IN");
    glCompileShader(id_);
    TI_TRACE("glCompileShader OUT");
    int status = GL_TRUE;
    glGetShaderiv(id_, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
      GLsizei logLength;
      glGetShaderiv(id_, GL_INFO_LOG_LENGTH, &logLength);
      auto log = std::vector<GLchar>(logLength + 1);
      glGetShaderInfoLog(id_, logLength, &logLength, log.data());
      log[logLength] = 0;
      TI_ERROR("[glsl] error while compiling shader:\n{}\n{}",
               add_line_markers(source), log.data());
    }
  }
};

struct GLProgram {
  GLuint id_;

  GLProgram() {
    id_ = glCreateProgram();
  }

  explicit GLProgram(GLuint id) : id_(id) {
  }

  explicit GLProgram(const GLShader &shader) : GLProgram() {
    this->attach(shader);
  }

  ~GLProgram() {
    glDeleteProgram(id_);
  }

  void attach(const GLShader &shader) const {
    glAttachShader(id_, shader.id_);
  }

  void link() const {
    TI_TRACE("glLinkProgram IN");
    glLinkProgram(id_);
    TI_TRACE("glLinkProgram OUT");
    int status = GL_TRUE;
    glGetProgramiv(id_, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
      GLsizei logLength;
      glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &logLength);
      auto log = std::vector<GLchar>(logLength + 1);
      glGetProgramInfoLog(id_, logLength, &logLength, log.data());
      log[logLength] = 0;
      TI_ERROR("[glsl] error while linking program:\n{}", log.data());
    }
  }

  void use() const {
    glUseProgram(id_);
  }
};

// https://blog.csdn.net/ylbs110/article/details/52074826
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
// This is Shader Storage Buffer, we use it to share data between CPU & GPU
class GLBuffer {
 private:
  GLuint id;
  GLenum type;
  size_t size;

 public:
  GLuint gl_get_id() const {
    return id;
  }

  GLuint gl_get_type() const {
    return type;
  }

  size_t get_size() const {
    return size;
  }

  GLBuffer(size_t size,
           void *initial_data = nullptr,
           GLenum type = GL_SHADER_STORAGE_BUFFER,
           GLbitfield access = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)
      : size(size), type(type) {
    glGenBuffers(1, &id);
    check_opengl_error("glGenBuffers");
    glBindBuffer(type, id);
    check_opengl_error("glBindBuffer");

    if (size) {
      glBufferStorage(type, size, initial_data, access);
      check_opengl_error("glBufferStorage");
    }
  }

  void bind_to_index(GLuint location, size_t offset, size_t size) {
    if (size) {
      glBindBufferRange(type, location, id, GLintptr(offset), GLsizeiptr(size));
      check_opengl_error("glBindBufferRange");
    }
  }

  void bind_to_index(GLuint location) {
    bind_to_index(location, /*offset=*/0, size);
  }

  void *map(GLenum access) {
    return map_region(/*offset=*/0, size, access);
  }

  void *map_region(size_t offset, size_t size, GLenum access) {
    glBindBuffer(type, id);
    check_opengl_error("glBindBuffer");
    void *mapped =
        glMapBufferRange(type, GLintptr(offset), GLsizeiptr(size), access);
    check_opengl_error("glMapBufferRange");
    TI_ASSERT(mapped);
    return mapped;
  }

  void unmap() {
    glBindBuffer(type, id);
    check_opengl_error("glBindBuffer");
    glUnmapBuffer(type);
    check_opengl_error("glUnmapBuffer");
  }

  void flush_mapped_region(size_t offset, size_t size) {
    glBindBuffer(type, id);
    check_opengl_error("glBindBuffer");
    glFlushMappedBufferRange(type, offset, size);
    check_opengl_error("glFlushMappedBufferRange");
  }

  void read_back(void *buffer, size_t offset, size_t size) {
    glBindBuffer(type, id);
    check_opengl_error("glBindBuffer");
    glGetBufferSubData(type, offset, size, buffer);
    check_opengl_error("glGetBufferSubData");
  }

  void read_back(void *buffer) {
    read_back(buffer, /*offset=*/0, size);
  }

  void upload(void *buffer, size_t offset, size_t size) {
    memcpy(map_region(offset, size, GL_MAP_WRITE_BIT), buffer, size);
    unmap();
  }

  void upload(void *buffer) {
    upload(buffer, /*offset=*/0, size);
  }
};

struct GLBufferAllocator {
 private:
  static constexpr size_t kMaxFreeResidentSize = 64 << 20;

  struct BufferKey {
    GLbitfield access{0};
    size_t size{0};

    struct Hash {
      size_t operator()(const BufferKey &k) const {
        return (size_t(k.access) << 48) ^ k.size;
      }
    };

    bool operator==(const BufferKey &k) const {
      return k.access == access && k.size == size;
    }
  };

  std::list<std::unique_ptr<GLBuffer>> buffers_storage_;

  std::unordered_multimap<BufferKey, GLBuffer *, BufferKey::Hash>
      buffers_mapping_;
  std::unordered_multimap<BufferKey, GLBuffer *, BufferKey::Hash> free_mapping_;

 public:
  GLBufferAllocator() {
  }

  void new_frame() {
    size_t free_resident_size = 0;
    for (auto pair : free_mapping_) {
      free_resident_size += pair.first.size;

      if (free_resident_size > kMaxFreeResidentSize) {
        GLBuffer *buf = pair.second;

        // TI_INFO("Release {}", (void *)buf);

        buffers_storage_.remove_if(
            [buf](const auto &p) { return p.get() == buf; });

        {
          auto buffer_iter =
              std::find_if(buffers_mapping_.begin(), buffers_mapping_.end(),
                           [buf](const auto &mo) { return mo.second == buf; });
          buffers_mapping_.erase(buffer_iter);
        }

        free_mapping_.erase(pair.first);
      }
    }
  }

  GLBuffer *alloc_buffer(size_t size,
                         void *base,
                         GLenum type = GL_SHADER_STORAGE_BUFFER,
                         GLbitfield access = GL_MAP_READ_BIT |
                                             GL_MAP_WRITE_BIT) {
    GLBuffer *buffer;
    auto buffer_iter = free_mapping_.find(BufferKey{access, size});
    if (buffer_iter == free_mapping_.end()) {
      // This buffer does not exist / can not be reused
      buffers_storage_.push_back(
          std::make_unique<GLBuffer>(size, base, type, access));
      buffer = buffers_storage_.back().get();
      buffers_mapping_.insert(std::pair(BufferKey{access, size}, buffer));

      // TI_INFO("New buffer {}, {}", size, (void *)buffer);
    } else {
      // Reuse
      buffer = buffer_iter->second;
      free_mapping_.erase(buffer_iter);

      if (base) {
        memcpy(buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT),
               base, size);
        buffer->unmap();
      } else {
        glInvalidateBufferData(buffer->gl_get_id());
      }

      // TI_INFO("Reuse buffer {}, {}", size, (void *)buffer);
    }

    return buffer;
  }

  void dealloc_buffer(GLBuffer *buf) {
    auto buffer_iter =
        std::find_if(buffers_mapping_.begin(), buffers_mapping_.end(),
                     [buf](const auto &mo) { return mo.second == buf; });
    if (buffer_iter != buffers_mapping_.end()) {
      // Insert back to free list
      free_mapping_.insert(std::pair(buffer_iter->first, buf));

      // TI_INFO("Dealloc {}", (void *)buf);
    }
  }
};

struct GLSLLauncherImpl {
  GLBufferAllocator gl_allocator;

  struct {
    GLBuffer *runtime = nullptr;
    GLBuffer *listman = nullptr;
    GLBuffer *root = nullptr;
    GLBuffer *gtmp = nullptr;
  } core_bufs;

  GLSLLauncherImpl() {
  }

  std::unique_ptr<GLSLRuntime> runtime;
  std::unique_ptr<GLSLListman> listman;

  std::vector<std::unique_ptr<CompiledProgram>> programs;
};

bool initialize_opengl(bool error_tolerance) {
  static std::optional<bool> supported;  // std::nullopt

  TI_TRACE("initialize_opengl({}) called", error_tolerance);

  if (supported.has_value()) {  // this function has been called before
    if (supported.value()) {    // detected to be true in last call
      return true;
    } else {
      if (!error_tolerance)  // not called from with_opengl
        TI_ERROR("OpenGL not supported");
      return false;
    }
  }

  glfwInit();
  // Compute Shader requires OpenGL 4.3+ (or OpenGL ES 3.1+)
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
  // GL context needs a window (There's no true headless GL)
  GLFWwindow *window =
      glfwCreateWindow(1, 1, "Make OpenGL Context", nullptr, nullptr);
  if (!window) {
    const char *desc = nullptr;
    int status = glfwGetError(&desc);
    if (!desc)
      desc = "Unknown Error";
    if (error_tolerance) {
      // error tolerated, returning false
      TI_DEBUG("[glsl] cannot create GLFW window: error {}: {}", status, desc);
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("[glsl] cannot create GLFW window: error {}: {}", status, desc);
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    if (error_tolerance) {
      TI_WARN("[glsl] cannot initialize GLAD");
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("[glsl] cannot initialize GLAD");
  }
#define PER_OPENGL_EXTENSION(x)          \
  if ((opengl_extension_##x = GLAD_##x)) \
    TI_TRACE("[glsl] Found " #x);
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION
  if (!opengl_extension_GL_ARB_compute_shader) {
    if (error_tolerance) {
      TI_INFO("Your OpenGL does not support GL_ARB_compute_shader extension");
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("Your OpenGL does not support GL_ARB_compute_shader extension");
  }

  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &opengl_max_block_dim);
  check_opengl_error("glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)");
  TI_TRACE("GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: {}", opengl_max_block_dim);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &opengl_max_grid_dim);
  check_opengl_error("glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_SIZE)");
  TI_TRACE("GL_MAX_COMPUTE_WORK_GROUP_SIZE: {}", opengl_max_grid_dim);

  supported = std::make_optional<bool>(true);
  return true;
}

void show_kernel_info(std::string const &kernel_name,
                      std::string const &kernel_source_code,
                      ParallelSize *ps) {
  bool is_accessor = taichi::starts_with(kernel_name, "snode_") ||
                     taichi::starts_with(kernel_name, "tensor_to_") ||
                     taichi::starts_with(kernel_name, "matrix_to_") ||
                     taichi::starts_with(kernel_name, "ext_arr_to_") ||
                     taichi::starts_with(kernel_name, "indirect_evaluator_") ||
                     taichi::starts_with(kernel_name, "jit_evaluator_");
  auto msg =
      fmt::format("[glsl]\ncompiling kernel {}<<<{}, {}>>>:\n{}", kernel_name,
                  ps->grid_dim, ps->block_dim, kernel_source_code);
  if (!is_accessor)
    TI_DEBUG("{}", msg);
  else
    TI_TRACE("{}", msg);
}

struct CompiledKernel::Impl {
  std::string kernel_name;
  std::unique_ptr<GLProgram> glsl;
  std::unique_ptr<ParallelSize> ps;
  std::string source;

  Impl(const std::string &kernel_name_,
       const std::string &kernel_source_code,
       std::unique_ptr<ParallelSize> ps_)
      : kernel_name(kernel_name_), ps(std::move(ps_)) {
    if (ps->grid_dim > opengl_max_grid_dim)
      ps->grid_dim = opengl_max_grid_dim;
    if (ps->block_dim > opengl_max_block_dim)
      ps->block_dim = opengl_max_block_dim;

    size_t layout_pos = kernel_source_code.find("precision highp float;\n");
    TI_ASSERT(layout_pos != std::string::npos);
    source = kernel_source_code.substr(0, layout_pos) +
             fmt::format(
                 "layout(local_size_x = {}, local_size_y = 1, local_size_z = "
                 "1) in;\n",
                 ps->block_dim) +
             kernel_source_code.substr(layout_pos);
    show_kernel_info(kernel_name_, source, ps.get());
    glsl = std::make_unique<GLProgram>(GLShader(source));
    glsl->link();
  }

  void dispatch_compute(GLSLLauncher *launcher) const {
    // https://www.khronos.org/opengl/wiki/Compute_Shader
    // https://community.arm.com/developer/tools-software/graphics/b/blog/posts/get-started-with-compute-shaders
    // https://www.khronos.org/assets/uploads/developers/library/2014-siggraph-bof/KITE-BOF_Aug14.pdf
    //
    // `glDispatchCompute(X, Y, Z)`   - the X*Y*Z  == `Blocks`   in CUDA
    // `layout(local_size_x = X) in;` - the X      == `Threads`  in CUDA
    //
    glsl->use();
    glDispatchCompute(ps->grid_dim, 1, 1);
    check_opengl_error("glDispatchCompute");

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    check_opengl_error("glMemoryBarrier");
  }
};

struct CompiledProgram::Impl {
  std::vector<std::unique_ptr<CompiledKernel>> kernels;
  int arg_count, ret_count;
  std::unordered_map<int, size_t> ext_arr_map;
  std::unordered_map<int, irpass::ExternalPtrAccess> ext_arr_access;
  std::vector<std::string> str_table;
  UsedFeature used;

  Impl(Kernel *kernel) {
    arg_count = kernel->args.size();
    ret_count = kernel->rets.size();
    for (int i = 0; i < arg_count; i++) {
      if (kernel->args[i].is_external_array) {
        ext_arr_map[i] = kernel->args[i].size;
      }
    }
  }

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           std::unique_ptr<ParallelSize> ps,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
    kernels.push_back(std::make_unique<CompiledKernel>(
        kernel_name, kernel_source_code, std::move(ps)));
    if (ext_ptr_access) {
      for (auto pair : *ext_ptr_access) {
        if (ext_arr_access.find(pair.first) != ext_arr_access.end()) {
          ext_arr_access[pair.first] = ext_arr_access[pair.first] | pair.second;
        } else {
          ext_arr_access[pair.first] = pair.second;
        }
      }
    }
  }

  int lookup_or_add_string(const std::string &str) {
    int i;
    for (i = 0; i < str_table.size(); i++) {
      if (str_table[i] == str) {
        return i;
      }
    }
    str_table.push_back(str);
    return i;
  }

  void dump_message_buffer(GLSLLauncher *launcher) const {
    auto runtime = launcher->impl->core_bufs.runtime;
    auto rt_buf = (GLSLRuntime *)runtime->map(GL_MAP_READ_BIT);

    auto msg_count = rt_buf->msg_count;
    if (msg_count > MAX_MESSAGES) {
      TI_WARN("[glsl] Too much print within one kernel: {} > {}, clipping",
              msg_count, MAX_MESSAGES);
      msg_count = MAX_MESSAGES;
    }

    for (int i = 0; i < msg_count; i++) {
      auto const &msg = rt_buf->msg_buf[i];
      for (int j = 0; j < msg.num_contents; j++) {
        int type = msg.get_type_of(j);
        auto value = msg.contents[j];

        std::string str;
        switch (type) {
          case 1:
            str = fmt::format("{}", value.val_i32);
            break;
          case 2:
            str = fmt::format("{}", value.val_f32);
            break;
          case 3:
            str = str_table.at(value.val_i32);
            break;
          default:
            TI_WARN("[glsl] Unexpected serialization type: {}, ignoring", type);
            break;
        };
        py_cout << str;
      }
    }
    rt_buf->msg_count = 0;
    runtime->unmap();
  }

  bool check_ext_arr_read(int i) const {
    auto iter = ext_arr_access.find(i);
    if (iter == ext_arr_access.end())
      return false;

    return (iter->second & irpass::ExternalPtrAccess::READ) !=
           irpass::ExternalPtrAccess::NONE;
  }

  bool check_ext_arr_write(int i) const {
    auto iter = ext_arr_access.find(i);
    if (iter == ext_arr_access.end())
      return false;

    return (iter->second & irpass::ExternalPtrAccess::WRITE) !=
           irpass::ExternalPtrAccess::NONE;
  }

  GLbitfield get_ext_arr_access(size_t &total_ext_arr_size) const {
    GLbitfield access = 0;
    for (const auto &[i, size] : ext_arr_map) {
      total_ext_arr_size += size;
      if (check_ext_arr_read(i)) {
        access |= GL_MAP_WRITE_BIT;
      }
      if (check_ext_arr_write(i)) {
        access |= GL_MAP_READ_BIT;
      }
    }
    return access;
  }

  void launch(Context &ctx, GLSLLauncher *launcher) const {
    launcher->impl->gl_allocator.new_frame();

    std::array<void *, taichi_max_num_args> ext_arr_host_ptrs;

    GLBuffer *extr_buf = nullptr;
    GLBuffer *args_buf = nullptr;
    GLBuffer *retr_buf = nullptr;
    uint8_t *args_buf_mapped = nullptr;

    // Prepare external array
    if (ext_arr_map.size()) {
      size_t total_ext_arr_size = 0;
      GLbitfield access = get_ext_arr_access(total_ext_arr_size);

      extr_buf = launcher->impl->gl_allocator.alloc_buffer(
          total_ext_arr_size, nullptr, GL_SHADER_STORAGE_BUFFER, access);

      void *baseptr = nullptr;
      if (access & GL_MAP_WRITE_BIT) {
        baseptr =
            extr_buf->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      }

      size_t accum_size = 0;
      for (const auto &[i, size] : ext_arr_map) {
        auto ptr = (void *)ctx.args[i];
        ctx.args[i] = accum_size;
        ext_arr_host_ptrs[i] = ptr;
        if (check_ext_arr_read(i)) {
          std::memcpy((char *)baseptr + accum_size, ptr, size);
        }
        accum_size += size;
      }

      if (baseptr)
        extr_buf->unmap();
    }

    // Prepare argument buffer
    {
      size_t args_buf_size = arg_count * sizeof(uint64_t);
      if (ext_arr_map.size()) {
        args_buf_size = taichi_opengl_earg_base +
                        arg_count * taichi_max_num_indices * sizeof(int);
      }

      if (args_buf_size > 0) {
        args_buf = launcher->impl->gl_allocator.alloc_buffer(
            args_buf_size, nullptr, GL_SHADER_STORAGE_BUFFER, GL_MAP_WRITE_BIT);

        args_buf_mapped = (uint8_t *)args_buf->map(
            GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        std::memcpy(args_buf_mapped, ctx.args, arg_count * sizeof(uint64_t));
        if (ext_arr_map.size()) {
          std::memcpy(args_buf_mapped + size_t(taichi_opengl_earg_base),
                      ctx.extra_args,
                      size_t(arg_count * taichi_max_num_indices) * sizeof(int));
        }
        args_buf->unmap();
      }
    }

    // Prepare return buffer
    if (ret_count > 0) {
      retr_buf = launcher->impl->gl_allocator.alloc_buffer(
          ret_count * sizeof(uint64_t), nullptr, GL_SHADER_STORAGE_BUFFER,
          GL_MAP_READ_BIT);
    }

    // Prepare runtime
    if (used.print) {
      // TODO(archibate): use result_buffer for print results
      auto runtime_buf = launcher->impl->core_bufs.runtime;
      auto mapped = (GLSLRuntime *)runtime_buf->map(GL_MAP_WRITE_BIT);
      mapped->msg_count = 0;
      runtime_buf->unmap();
    }

    // Bind uniforms (descriptors in low level APIs)
    auto &core_bufs = launcher->impl->core_bufs;
    core_bufs.runtime->bind_to_index(GLuint(GLBufId::Runtime));
    core_bufs.listman->bind_to_index(GLuint(GLBufId::Listman));
    core_bufs.root->bind_to_index(GLuint(GLBufId::Root));
    core_bufs.gtmp->bind_to_index(GLuint(GLBufId::Gtmp));

    if (args_buf)
      args_buf->bind_to_index(GLuint(GLBufId::Args));
    if (retr_buf)
      retr_buf->bind_to_index(GLuint(GLBufId::Retr));
    if (extr_buf)
      extr_buf->bind_to_index(GLuint(GLBufId::Extr));

    // Kernel dispatch
    for (const auto &ker : kernels) {
      ker->dispatch_compute(launcher);
    }

    // Data read-back
    if (used.print) {
      dump_message_buffer(launcher);
    }

    if (extr_buf) {
      for (const auto &[i, size] : ext_arr_map) {
        if (check_ext_arr_write(i)) {
          extr_buf->read_back(ext_arr_host_ptrs[i], size_t(ctx.args[i]), size);
        }
      }
    }

    if (ret_count > 0) {
      retr_buf->read_back(launcher->result_buffer, 0,
                          ret_count * sizeof(uint64_t));
    }

    if (args_buf)
      launcher->impl->gl_allocator.dealloc_buffer(args_buf);
    if (retr_buf)
      launcher->impl->gl_allocator.dealloc_buffer(retr_buf);
    if (extr_buf)
      launcher->impl->gl_allocator.dealloc_buffer(extr_buf);
  }
};

GLSLLauncher::GLSLLauncher(size_t root_size) {
  initialize_opengl();
  impl = std::make_unique<GLSLLauncherImpl>();

  impl->runtime = std::make_unique<GLSLRuntime>();
  impl->core_bufs.runtime =
      impl->gl_allocator.alloc_buffer(sizeof(GLSLRuntime), impl->runtime.get());

  impl->listman = std::make_unique<GLSLListman>();
  impl->core_bufs.listman =
      impl->gl_allocator.alloc_buffer(sizeof(GLSLListman), impl->listman.get());

  impl->core_bufs.root = impl->gl_allocator.alloc_buffer(
      root_size, nullptr, GL_SHADER_STORAGE_BUFFER, 0);

  impl->core_bufs.gtmp = impl->gl_allocator.alloc_buffer(
      taichi_global_tmp_buffer_size, nullptr, GL_SHADER_STORAGE_BUFFER, 0);
}

void GLSLLauncher::keep(std::unique_ptr<CompiledProgram> program) {
  impl->programs.push_back(std::move(program));
}

bool is_opengl_api_available() {
  if (get_environ_config("TI_ENABLE_OPENGL", 1) == 0)
    return false;
  return initialize_opengl(true);
}

#else
struct GLProgram {};
struct GLSLLauncherImpl {};

struct CompiledKernel::Impl {
  Impl(const std::string &kernel_name_,
       const std::string &kernel_source_code,
       std::unique_ptr<ParallelSize> ps_) {
    TI_NOT_IMPLEMENTED;
  }

  void dispatch_compute(GLSLLauncher *launcher) const {
    TI_NOT_IMPLEMENTED;
  }
};

struct CompiledProgram::Impl {
  UsedFeature used;

  Impl(Kernel *kernel) {
    TI_NOT_IMPLEMENTED;
  }

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           std::unique_ptr<ParallelSize> ps,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
    TI_NOT_IMPLEMENTED;
  }

  int lookup_or_add_string(const std::string &str) {
    TI_NOT_IMPLEMENTED;
  }

  void launch(Context &ctx, GLSLLauncher *launcher) const {
    TI_NOT_IMPLEMENTED;
  }
};

GLSLLauncher::GLSLLauncher(size_t size) {
  TI_NOT_IMPLEMENTED;
}

void GLSLLauncher::keep(std::unique_ptr<CompiledProgram>) {
  TI_NOT_IMPLEMENTED;
}

bool is_opengl_api_available() {
  return false;
}

bool initialize_opengl(bool error_tolerance) {
  TI_NOT_IMPLEMENTED;
}

#endif  // TI_WITH_OPENGL

CompiledProgram::CompiledProgram(Kernel *kernel)
    : impl(std::make_unique<Impl>(kernel)) {
}

CompiledProgram::~CompiledProgram() = default;

void CompiledProgram::add(
    const std::string &kernel_name,
    const std::string &kernel_source_code,
    std::unique_ptr<ParallelSize> ps,
    std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
  impl->add(kernel_name, kernel_source_code, std::move(ps), ext_ptr_access);
}

void CompiledProgram::set_used(const UsedFeature &used) {
  impl->used = used;
}

int CompiledProgram::lookup_or_add_string(const std::string &str) {
  return impl->lookup_or_add_string(str);
}

void CompiledProgram::launch(Context &ctx, GLSLLauncher *launcher) const {
  impl->launch(ctx, launcher);
}

CompiledKernel::CompiledKernel(const std::string &kernel_name_,
                               const std::string &kernel_source_code,
                               std::unique_ptr<ParallelSize> ps_)
    : impl(std::make_unique<Impl>(kernel_name_,
                                  kernel_source_code,
                                  std::move(ps_))) {
}

void CompiledKernel::dispatch_compute(GLSLLauncher *launcher) const {
  impl->dispatch_compute(launcher);
}

CompiledKernel::~CompiledKernel() = default;

GLSLLauncher::~GLSLLauncher() = default;

}  // namespace opengl
TLANG_NAMESPACE_END
