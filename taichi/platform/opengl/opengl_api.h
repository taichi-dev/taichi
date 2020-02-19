#pragma once

#include <taichi/common.h>
#include <taichi/common/util.h>

#include <string>
#include <vector>

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct IOV
{
  void *base;
  size_t size;
};

bool is_opengl_api_available();
std::vector<void *> launch_glsl_kernel(std::string source, std::vector<IOV> iov);
void unmap_all_ssbo();

}  // namespace opengl

TLANG_NAMESPACE_END
