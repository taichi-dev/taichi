#pragma once

#include "taichi/common/util.h"

#include <string>
#include <vector>

#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct GLProgram;
void initialize_opengl();
bool is_opengl_api_available();
void create_glsl_root_buffer(size_t size);
void begin_glsl_kernels(const std::vector<IOV> &iov);
void launch_glsl_kernel(GLProgram *program, int num_groups);
void end_glsl_kernels(const std::vector<IOV> &iov);
GLProgram *compile_glsl_program(std::string source);

}  // namespace opengl

TLANG_NAMESPACE_END
