#pragma once

#include "taichi/common/util.h"

#include <string>
#include <vector>

#include "opengl_kernel_launcher.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

void initialize_opengl();
bool is_opengl_api_available();
int opengl_get_threads_per_group();
extern bool opengl_has_GL_NV_shader_atomic_float;

struct GLProgram;
struct CompiledGLSL {
  std::unique_ptr<GLProgram> glsl;
  CompiledGLSL(const std::string &source);
  ~CompiledGLSL();
  void launch_glsl(int num_groups) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
