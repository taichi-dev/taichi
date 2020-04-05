#pragma once

#include "taichi/common/util.h"

#include <string>
#include <vector>

#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

void initialize_opengl();
bool is_opengl_api_available();
void create_glsl_root_buffer(size_t size);
void begin_glsl_kernels(const std::vector<IOV> &iov);
void end_glsl_kernels(const std::vector<IOV> &iov);
int opengl_get_threads_per_group();
extern bool opengl_has_GL_NV_shader_atomic_float;

struct GLProgram;
struct CompiledGLSL {
  std::unique_ptr<GLProgram> glsl;
  CompiledGLSL(std::string source);
  ~CompiledGLSL();
  void launch_glsl(int num_groups) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
