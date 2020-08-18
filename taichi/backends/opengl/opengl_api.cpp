//#define _GLSL_DEBUG 1
#include "opengl_api.h"

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/util/environ_config.h"
#include "taichi/python/print_buffer.h"
#include "taichi/backends/opengl/shaders/runtime.h"
#include "taichi/backends/opengl/shaders/listman.h"

#ifdef TI_WITH_OPENGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#endif

TLANG_NAMESPACE_BEGIN
namespace opengl {

#define PER_OPENGL_EXTENSION(x) bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

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

int opengl_get_threads_per_group() {
  int ret = 1000;
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &ret);
  check_opengl_error("glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)");
  TI_TRACE("opengl_get_threads_per_group: {}", ret);
  return ret;
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
struct GLSSBO {
  GLuint id_;

  GLSSBO() {
    glGenBuffers(1, &id_);
  }

  ~GLSSBO() {
    glDeleteBuffers(1, &id_);
  }

  /***
   GL_{frequency}_{nature}:


   STREAM
       The data store contents will be modified once and used at most a few
   times.

   STATIC
       The data store contents will be modified once and used many times.

   DYNAMIC
       The data store contents will be modified repeatedly and used many times.


   DRAW
       The data store contents are modified by the application, and used as the
   source for GL drawing and image specification commands.

   READ
       The data store contents are modified by reading data from the GL, and
   used to return that data when queried by the application.

   COPY
       The data store contents are modified by reading data from the GL, and
   used as the source for GL drawing and image specification commands.
   ***/

  void bind_data(void *data,
                 size_t size,
                 GLuint usage = GL_DYNAMIC_READ) const {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    check_opengl_error("glBindBuffer");
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage);
    check_opengl_error("glBufferData");
  }

  void bind_index(size_t index) const {
    // SSBO index, is `layout(std430, binding = <index>)` in shader.
    // We use only one SSBO though...
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, id_);
    check_opengl_error("glBindBufferBase");
  }

  void bind_range(size_t index, size_t offset, size_t size) const {
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, index, id_, offset, size);
    check_opengl_error("glBindBufferRange");
  }

  void *map(size_t offset,
            size_t length,
            GLbitfield access = GL_READ_ONLY) const {
    // map GPU memory to CPU address space, offset within SSBO data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    check_opengl_error("glBindBuffer");
    void *p =
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset, length, access);
    check_opengl_error("glMapBufferRange");
    TI_ASSERT_INFO(p, "glMapBufferRange returned NULL");
    return p;
  }

  void *map(GLbitfield access = GL_READ_ONLY) const {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    check_opengl_error("glBindBuffer");
    void *p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, access);
    check_opengl_error("glMapBuffer");
    TI_ASSERT_INFO(p, "glMapBuffer returned NULL");
    return p;
  }

  void unmap() const {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    check_opengl_error("glBindBuffer");
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    check_opengl_error("glUnmapBuffer");
  }
};

struct GLBuffer : GLSSBO {
  GLBufId index;
  void *base;
  size_t size;

  GLBuffer(GLBufId index, void *base, size_t size)
      : index(index), base(base), size(size) {
    bind_data(base, size);
    bind_index((int)index);
  }

  GLBuffer(GLBufId index) : index(index), base(nullptr), size(0) {
    bind_index((int)index);
  }

  void copy_forward() {
    bind_data(base, size);
  }

  void rebind(void *new_base, size_t new_size) {
    base = new_base;
    size = new_size;
    bind_data(base, size);
  }

  void copy_back() {
    if (!size)
      return;
    void *mapped = this->map();
    TI_ASSERT(mapped);
    std::memcpy(base, mapped, size);
    this->unmap();
  }
};

struct GLBufferTable {
  std::map<GLBufId, std::unique_ptr<GLBuffer>> bufs;

  GLBuffer *get(GLBufId id) {
    return bufs.at(id).get();
  }

  void add_buffer(GLBufId index, void *base, size_t size) {
    bufs[index] = std::make_unique<GLBuffer>(index, base, size);
  }

  void clear() {
    bufs.clear();
  }
};

struct GLSLLauncherImpl {
  GLBufferTable core_bufs;
  GLBufferTable user_bufs;

  std::vector<char> root_buffer;
  std::vector<char> gtmp_buffer;
  std::unique_ptr<GLSLRuntime> runtime;
  std::unique_ptr<GLSLListman> listman;

  std::vector<std::unique_ptr<CompiledProgram>> programs;
};

ParallelSize::~ParallelSize() {
}

size_t ParallelSize::get_threads_per_block() const {
  size_t limit = opengl_get_threads_per_group();
  size_t n = threads_per_block.value_or(0);
  return n == 0 ? limit : std::min(n, limit);
}

ParallelSize_ConstRange::ParallelSize_ConstRange(size_t num_strides)
    : num_strides(num_strides) {
}

size_t ParallelSize_ConstRange::get_num_strides(GLSLLauncher *launcher) const {
  return num_strides;
}

size_t ParallelSize_ConstRange::get_threads_per_block() const {
  size_t n = get_num_threads(nullptr);
  size_t TPG = ParallelSize::get_threads_per_block();
  return std::max(std::min(n, TPG), (size_t)1);
}

size_t ParallelSize_DynamicRange::get_num_strides(
    GLSLLauncher *launcher) const {
  size_t n;
  auto gtmp = launcher->impl->core_bufs.get(GLBufId::Gtmp);
  void *gtmp_now = gtmp->map();  // TODO: RAII map/unmap
  size_t b = range_begin, e = range_end;
  if (!const_begin)
    b = *(const int *)((const char *)gtmp_now + b);
  if (!const_end)
    e = *(const int *)((const char *)gtmp_now + e);
  gtmp->unmap();
  if (e <= b)
    n = 1;
  else
    n = e - b;
  return n;
}

size_t ParallelSize_StructFor::get_num_strides(GLSLLauncher *launcher) const {
  auto listman = launcher->impl->core_bufs.get(GLBufId::Listman);
  auto lm_buf = (GLSLListman *)listman->map();
  auto n = lm_buf->list_len;
  listman->unmap();
  return n;
}

size_t ParallelSize::get_num_threads(GLSLLauncher *launcher) const {
  size_t n = get_num_strides(launcher);
  size_t SPT = strides_per_thread.value_or(1);
  return std::max((n + SPT - 1) / SPT, (size_t)1);
}

size_t ParallelSize::get_num_blocks(GLSLLauncher *launcher) const {
  size_t n = get_num_threads(launcher);
  size_t TPG = get_threads_per_block();
  return std::max((n + TPG - 1) / TPG, (size_t)1);
}

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
  // GLEW cannot load GL without a context
  // And the best way to make context is by creating a window
  // Then hide it immediately, LOL
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
  glfwHideWindow(window);
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

  supported = std::make_optional<bool>(true);
  return true;
}

void display_kernel_info(std::string const &kernel_name,
                         std::string const &kernel_source_code) {
  bool is_accessor = taichi::starts_with(kernel_name, "snode_") ||
                     taichi::starts_with(kernel_name, "tensor_to_") ||
                     taichi::starts_with(kernel_name, "matrix_to_") ||
                     taichi::starts_with(kernel_name, "ext_arr_to_") ||
                     taichi::starts_with(kernel_name, "jit_evaluator_");
  if (!is_accessor)
    TI_DEBUG("source of kernel [{}]:\n{}", kernel_name, kernel_source_code);
  else
    TI_TRACE("source of kernel [{}]:\n{}", kernel_name, kernel_source_code);
#ifdef _GLSL_DEBUG
  std::ofstream(fmt::format("/tmp/{}.comp", kernel_name))
      .write(kernel_source_code.c_str(), kernel_source_code.size());
#endif
}

struct CompiledKernel {
  std::string kernel_name;
  std::unique_ptr<GLProgram> glsl;
  std::unique_ptr<ParallelSize> ps;
  std::string source;

  // disscussion:
  // https://github.com/taichi-dev/taichi/pull/696#issuecomment-609332527
  CompiledKernel(CompiledKernel &&) = default;
  CompiledKernel &operator=(CompiledKernel &&) = default;

  explicit CompiledKernel(const std::string &kernel_name_,
                          const std::string &kernel_source_code,
                          std::unique_ptr<ParallelSize> ps_)
      : kernel_name(kernel_name_), ps(std::move(ps_)) {
    source =
        kernel_source_code +
        fmt::format(
            "layout(local_size_x = {}, local_size_y = 1, local_size_z = 1) in;",
            ps->get_threads_per_block());
    display_kernel_info(kernel_name_, source);
    glsl = std::make_unique<GLProgram>(GLShader(source));
    glsl->link();
  }

  void dispatch_compute(GLSLLauncher *launcher) const {
    int num_blocks = ps->get_num_blocks(launcher);

    glsl->use();

    // https://www.khronos.org/opengl/wiki/Compute_Shader
    // https://community.arm.com/developer/tools-software/graphics/b/blog/posts/get-started-with-compute-shaders
    // https://www.khronos.org/assets/uploads/developers/library/2014-siggraph-bof/KITE-BOF_Aug14.pdf
    //
    // `glDispatchCompute(X, Y, Z)`   - the X*Y*Z  == `Blocks`   in CUDA
    // `layout(local_size_x = X) in;` - the X      == `Threads`  in CUDA
    //
    glDispatchCompute(num_blocks, 1, 1);
    check_opengl_error(fmt::format("glDispatchCompute({})", num_blocks));

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    check_opengl_error("glMemoryBarrier");
  }
};

struct CompiledProgram::Impl {
  std::vector<std::unique_ptr<CompiledKernel>> kernels;
  int arg_count, ret_count;
  std::map<int, size_t> ext_arr_map;
  std::vector<std::string> str_table;
  UsedFeature used;

  Impl(Kernel *kernel) {
    arg_count = kernel->args.size();
    ret_count = kernel->rets.size();
    for (int i = 0; i < arg_count; i++) {
      if (kernel->args[i].is_nparray) {
        ext_arr_map[i] = kernel->args[i].size;
      }
    }
  }

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           std::unique_ptr<ParallelSize> ps) {
    kernels.push_back(std::make_unique<CompiledKernel>(
        kernel_name, kernel_source_code, std::move(ps)));
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
    auto runtime = launcher->impl->core_bufs.get(GLBufId::Runtime);
    auto rt_buf = (GLSLRuntime *)runtime->map();

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

  void launch(Context &ctx, GLSLLauncher *launcher) const {
    GLBufferTable &bufs = launcher->impl->user_bufs;
    std::vector<char> base_arr;
    std::vector<void *> saved_ctx_ptrs;
    // NOTE: these dirty codes are introduced by #694, TODO: RAII
    /// DIRTY_BEGIN {{{
    if (ext_arr_map.size()) {
      bufs.add_buffer(GLBufId::Earg, ctx.extra_args,
                      arg_count * taichi_max_num_args * sizeof(int));
      if (ext_arr_map.size() == 1) {  // zero-copy for only one ext_arr
        auto it = ext_arr_map.begin();
        auto extptr = (void *)ctx.args[it->first];
        ctx.args[it->first] = 0;
        bufs.add_buffer(GLBufId::Extr, extptr, it->second);
      } else {
        size_t accum_size = 0;
        std::vector<void *> ptrarr;
        for (const auto &[i, size] : ext_arr_map) {
          accum_size += size;
        }
        base_arr.resize(accum_size);
        void *baseptr = base_arr.data();
        accum_size = 0;
        for (const auto &[i, size] : ext_arr_map) {
          auto ptr = (void *)ctx.args[i];
          saved_ctx_ptrs.push_back(ptr);
          std::memcpy((char *)baseptr + accum_size, ptr, size);
          ctx.args[i] = accum_size;
          accum_size += size;
        }  // concat all extptr into my baseptr
        bufs.add_buffer(GLBufId::Extr, baseptr, accum_size);
      }
    }
    /// DIRTY_END }}}
    bufs.add_buffer(GLBufId::Args, ctx.args,
                    std::max(arg_count, ret_count) * sizeof(uint64_t));
    if (used.print) {
      auto runtime_buf = launcher->impl->core_bufs.get(GLBufId::Runtime);
      auto mapped = (GLSLRuntime *)runtime_buf->map();
      mapped->msg_count = 0;
      runtime_buf->unmap();
    }
    for (const auto &ker : kernels) {
      ker->dispatch_compute(launcher);
    }
    for (auto &[idx, buf] : launcher->impl->user_bufs.bufs) {
      buf->copy_back();
    }
    launcher->impl->user_bufs.clear();
    if (used.print) {
      dump_message_buffer(launcher);
    }
    /// DIRTY_BEGIN {{{
    if (ext_arr_map.size() > 1) {
      void *baseptr = base_arr.data();
      auto cpit = saved_ctx_ptrs.begin();
      size_t accum_size = 0;
      for (const auto &[i, size] : ext_arr_map) {
        std::memcpy(*cpit, (char *)baseptr + accum_size, size);
        accum_size += size;
        cpit++;
      }  // extract back to all extptr from my baseptr
    }
    /// DIRTY_END }}}
  }
};

GLSLLauncher::GLSLLauncher(size_t root_size) {
  initialize_opengl();
  impl = std::make_unique<GLSLLauncherImpl>();

  impl->runtime = std::make_unique<GLSLRuntime>();
  impl->core_bufs.add_buffer(GLBufId::Runtime, impl->runtime.get(),
                             sizeof(GLSLRuntime));

  impl->listman = std::make_unique<GLSLListman>();
  impl->core_bufs.add_buffer(GLBufId::Listman, impl->listman.get(),
                             sizeof(GLSLListman));

  impl->root_buffer.resize(root_size, 0);
  impl->core_bufs.add_buffer(GLBufId::Root, impl->root_buffer.data(),
                             root_size);

  impl->gtmp_buffer.resize(taichi_global_tmp_buffer_size, 0);
  impl->core_bufs.add_buffer(GLBufId::Gtmp, impl->gtmp_buffer.data(),
                             taichi_global_tmp_buffer_size);
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

struct CompiledProgram::Impl {
  UsedFeature used;

  Impl(Kernel *kernel) {
    TI_NOT_IMPLEMENTED;
  }

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           std::unique_ptr<ParallelSize> ps) {
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

ParallelSize::~ParallelSize() {
}

size_t ParallelSize::get_num_threads(GLSLLauncher *launcher) const {
  TI_NOT_IMPLEMENTED;
}

size_t ParallelSize::get_num_blocks(GLSLLauncher *launcher) const {
  TI_NOT_IMPLEMENTED;
}

size_t ParallelSize::get_threads_per_block() const {
  TI_NOT_IMPLEMENTED;
}

size_t ParallelSize_ConstRange::get_threads_per_block() const {
  TI_NOT_IMPLEMENTED;
}

size_t ParallelSize_ConstRange::get_num_strides(GLSLLauncher *launcher) const {
  TI_NOT_IMPLEMENTED;
}

size_t ParallelSize_DynamicRange::get_num_strides(
    GLSLLauncher *launcher) const {
  TI_NOT_IMPLEMENTED;
}

size_t ParallelSize_StructFor::get_num_strides(GLSLLauncher *launcher) const {
  TI_NOT_IMPLEMENTED;
}

ParallelSize_ConstRange::ParallelSize_ConstRange(size_t num_strides)
    : num_strides(num_strides) {
}

#endif  // TI_WITH_OPENGL

CompiledProgram::CompiledProgram(Kernel *kernel)
    : impl(std::make_unique<Impl>(kernel)) {
}

CompiledProgram::~CompiledProgram() = default;

void CompiledProgram::add(const std::string &kernel_name,
                          const std::string &kernel_source_code,
                          std::unique_ptr<ParallelSize> ps) {
  impl->add(kernel_name, kernel_source_code, std::move(ps));
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

GLSLLauncher::~GLSLLauncher() = default;

}  // namespace opengl
TLANG_NAMESPACE_END
