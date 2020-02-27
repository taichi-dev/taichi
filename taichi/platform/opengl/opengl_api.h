#pragma once

#include <taichi/common.h>
#include <taichi/common/util.h>

#include <string>
#include <vector>

#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

void initialize_opengl();
bool is_opengl_api_available();
void launch_glsl_kernel(std::string source, std::vector<IOV> iov);

}  // namespace opengl

TLANG_NAMESPACE_END
