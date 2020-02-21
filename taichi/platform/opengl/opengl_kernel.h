#pragma once

#include <taichi/common.h>
#include <taichi/constants.h>
#include <taichi/tlang_util.h>

#include <string>
#include <vector>

#include "opengl_kernel_util.h"


TLANG_NAMESPACE_BEGIN

namespace opengl {

struct SSBO
{
  void *const data;
  const size_t data_size;

  SSBO(size_t data_size);
  ~SSBO();

  void load_from(const void *buffer);
  void save_to(void *buffer);

  operator IOV()
  {
    return IOV{data, data_size};
  }
};

}  // namespace opengl

TLANG_NAMESPACE_END
