#pragma once

#include <taichi/inc/constants.h>
#include <taichi/lang_util.h>

#include <string>
#include <vector>

#include "opengl_kernel_util.h"


TLANG_NAMESPACE_BEGIN

namespace opengl {

struct SSBO
{
  std::vector<uint8_t> data_;
  const size_t data_size;

  SSBO(size_t data_size);

  void load_arguments_from(Context &ctx);
  void save_returns_to(Context &ctx);
  inline void *data()
  {
    return (void *)data_.data();
  }

  inline operator IOV()
  {
    return IOV{data(), data_size};
  }
};

}  // namespace opengl

TLANG_NAMESPACE_END
