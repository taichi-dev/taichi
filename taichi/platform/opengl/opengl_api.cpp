#include "opengl_api.h"
#include <taichi/perf.h>
#include <taichi/program/program.h>

#ifdef TI_WITH_OPENGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

TLANG_NAMESPACE_BEGIN
namespace opengl {

#ifdef TI_WITH_OPENGL
void glapi_set_uniform(GLuint loc, float value)
{
  glUniform1f(loc, value);
}

static std::string add_line_markers(std::string x)
{
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

struct GLShader
{
  GLuint id_;

  GLShader(GLuint type = GL_COMPUTE_SHADER)
  {
    id_ = glCreateShader(type);
  }

  GLShader(std::string source, GLuint type = GL_COMPUTE_SHADER)
    : GLShader(type)
  {
    this->compile(source);
  }

  ~GLShader()
  {
    glDeleteShader(id_);
  }

  GLShader &compile(const std::string &source)
  {
    const GLchar *source_cstr = source.c_str();
    glShaderSource(id_, 1, &source_cstr, nullptr);

    glCompileShader(id_);
    GLint status = GL_TRUE;
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
    return *this;
  }
};

struct GLProgram
{
  GLuint id_;

  GLProgram()
  {
    id_ = glCreateProgram();
  }

  explicit GLProgram(GLuint id)
    : id_(id)
  {}

  explicit GLProgram(GLShader &shader)
    : GLProgram()
  {
    this->attach(shader);
  }

  ~GLProgram()
  {
    glDeleteProgram(id_);
  }

  GLProgram &attach(GLShader &shader)
  {
    glAttachShader(id_, shader.id_);
    return *this;
  }

  GLProgram &link()
  {
    glLinkProgram(id_);
    GLint status = GL_TRUE;
    glGetProgramiv(id_, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
      GLsizei logLength;
      glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &logLength);
      auto log = std::vector<GLchar>(logLength + 1);
      glGetProgramInfoLog(id_, logLength, &logLength, log.data());
      log[logLength] = 0;
      TI_ERROR("[glsl] error while linking program:\n{}", log.data());
    }
    return *this;
  }

  GLProgram &use()
  {
    glUseProgram(id_);
    return *this;
  }

  template <typename T>
  void set_uniform(std::string name, T value)
  {
    GLuint loc = glGetUniformLocation(id_, name.c_str());
    glapi_set_uniform(loc, value);
  }
};


// https://blog.csdn.net/ylbs110/article/details/52074826
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
// This is Shader Storage Buffer, we use it to share data between CPU & GPU
struct GLSSBO
{
  GLuint id_;

  GLSSBO()
  {
    glGenBuffers(1, &id_);
  }

  ~GLSSBO()
  {
    glDeleteBuffers(1, &id_);
  }

  /***
   GL_{frequency}_{nature}:


   STREAM
       The data store contents will be modified once and used at most a few times.

   STATIC
       The data store contents will be modified once and used many times.

   DYNAMIC
       The data store contents will be modified repeatedly and used many times.


   DRAW
       The data store contents are modified by the application, and used as the source
       for GL drawing and image specification commands.

   READ
       The data store contents are modified by reading data from the GL, and used to
       return that data when queried by the application.

   COPY
       The data store contents are modified by reading data from the GL, and used as the
       source for GL drawing and image specification commands.
   ***/

  GLSSBO &bind_data(void *data, size_t size, GLuint usage = GL_STATIC_READ)
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage);
    return *this;
  }

  GLSSBO &bind_index(size_t index)
  {
    // SSBO index, is `layout(std430, binding = <index>)` in shader.
    // We use only one SSBO though...
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, id_);
    return *this;
  }

  GLSSBO &bind_range(size_t index, size_t offset, size_t size)
  {
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, index, id_, offset, size);
    return *this;
  }

  void *map(size_t offset, size_t length, GLbitfield access = GL_MAP_READ_BIT)
  {
    // map GPU memory to CPU address space, offset within SSBO data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    return glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset, length, access);
  }

  void *map(GLbitfield access = GL_MAP_READ_BIT)
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_);
    return glMapBuffer(GL_SHADER_STORAGE_BUFFER, access);
  }
};

void initialize_opengl()
{
  static bool gl_inited = false;
  if (gl_inited)
    return;
  TI_WARN("OpenGL backend currently WIP, MAY NOT WORK");
  gl_inited = true;

  glfwInit();
  // Compute Shader requires OpenGL 4.3+ (or OpenGL ES 3.1+)
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  // GLEW cannot load GL without a context
  // And the best way to make context is by creating a window
  // Then hide it immediately, LOL
  GLFWwindow *window = glfwCreateWindow(1, 1, "Make GLEW Happy", nullptr, nullptr);
  if (!window) {
    const char *desc = nullptr;
    GLint status = glfwGetError(&desc);
    if (!desc) desc = "Unknown Error";
    TI_ERROR("[glsl] cannot create GLFW window: error {}: {}", status, desc);
  }
  glfwHideWindow(window);
  glfwMakeContextCurrent(window);
  GLint status = glewInit();
  if (status != GLEW_OK) {
    TI_ERROR("[glsl] cannot initialize GLEW: {}", glewGetErrorString(status));
  }
  const char *gl_version = (const char *)glGetString(GL_VERSION);
  if (!gl_version) {
    TI_WARN("[glsl] cannot get OpenGL version");
  } else {
    TI_INFO("[glsl] OpenGL {}", gl_version);
  }
}

GLProgram *compile_glsl_program(std::string source)
{
  GLShader shader(source);
  GLProgram *program = new GLProgram(shader);
  program->link();
  return program;
}

GLSSBO *root_ssbo;

void create_glsl_root_buffer(size_t size)
{
  //if (root_ssbo) return;
  initialize_opengl();
  root_ssbo = new GLSSBO; // TODO(archibate): mem leaking
  void *buffer = std::calloc(size, 1);
  root_ssbo->bind_data(buffer, size, GL_DYNAMIC_READ);
  root_ssbo->bind_index(0);
}

std::vector<GLSSBO> ssbo;

void begin_glsl_kernels(const std::vector<IOV> &iov)
{
  ssbo = std::vector<GLSSBO>(iov.size());

  for (int i = 0; i < ssbo.size(); i++) {
    if (!iov[i].size)
      continue;
    ssbo[i].bind_index(i + 1);
    ssbo[i].bind_data(iov[i].base, iov[i].size, GL_DYNAMIC_READ); // input
  }
}

void launch_glsl_kernel(GLProgram *program, int num_groups)
{
  program->use();

  // https://www.khronos.org/opengl/wiki/Compute_Shader
  // https://community.arm.com/developer/tools-software/graphics/b/blog/posts/get-started-with-compute-shaders
  // https://www.khronos.org/assets/uploads/developers/library/2014-siggraph-bof/KITE-BOF_Aug14.pdf
  //
  // `glDispatchCompute(X, Y, Z)`   - the X*Y*Z  == `Blocks`   in CUDA
  // `layout(local_size_x = X) in;` - the X      == `Threads`  in CUDA
  //
  glDispatchCompute(num_groups, 1, 1);
}

void end_glsl_kernels(const std::vector<IOV> &iov)
{
  //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // TODO(archibate): move to Program::synchroize()

  for (int i = 0; i < ssbo.size(); i++) {
    if (!iov[i].size)
      continue;
    void *p = ssbo[i].map(0, iov[i].size); // output
    std::memcpy(iov[i].base, p, iov[i].size);
  }
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  ssbo.clear();
}

bool is_opengl_api_available()
{
  return true;
}

#else
void create_glsl_root_buffer(size_t size)
{
  TI_NOT_IMPLEMENTED
}

void begin_glsl_kernels(const std::vector<IOV> &iov)
{
  TI_NOT_IMPLEMENTED
}

void end_glsl_kernels(const std::vector<IOV> &iov)
{
  TI_NOT_IMPLEMENTED
}

void launch_glsl_kernel(GLProgram *program, int num_groups)
{
  TI_NOT_IMPLEMENTED
}

bool is_opengl_api_available()
{
  return false;
}

void initialize_opengl()
{
}

GLProgram *compile_glsl_program(std::string source)
{
  return nullptr;
}
#endif

}
TLANG_NAMESPACE_END
