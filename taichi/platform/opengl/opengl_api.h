#pragma once

#include <taichi/common.h>
#include <taichi/common/util.h>

#include <string>
#include <vector>

TLANG_NAMESPACE_BEGIN

namespace opengl {

bool is_opengl_api_available();
void *launch_glsl_kernel(std::string source, void *data, size_t data_size);

}  // namespace opengl

TLANG_NAMESPACE_END
