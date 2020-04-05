#pragma once

#include "taichi/common/util.h"

#include <string>
#include <vector>

#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

void initialize_opengl();
bool is_opengl_api_available();
int opengl_get_threads_per_group();
extern bool opengl_has_GL_NV_shader_atomic_float;

struct GLProgram;
struct CompiledGLSL {
  std::unique_ptr<GLProgram> glsl;
  CompiledGLSL(std::string source);
  ~CompiledGLSL();
  void launch_glsl(int num_groups) const;
};

struct GLSLLauncherImpl;
struct GLSLLauncher {
  std::unique_ptr<GLSLLauncherImpl> impl;
  GLSLLauncher(size_t size);
  ~GLSLLauncher();
  void begin_glsl_kernels(const std::vector<IOV> &iov);
  void end_glsl_kernels(const std::vector<IOV> &iov);
};

}  // namespace opengl

TLANG_NAMESPACE_END
