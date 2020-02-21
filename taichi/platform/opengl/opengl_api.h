#pragma once

#include <taichi/common.h>
#include <taichi/common/util.h>

#include <string>
#include <vector>

#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct GLProgram;
bool is_opengl_api_available();
void launch_glsl_kernel(GLProgram *program, std::vector<IOV> iov, int num_groups);
GLProgram *compile_glsl_program(std::string source);

}  // namespace opengl

TLANG_NAMESPACE_END
