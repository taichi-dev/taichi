#pragma once

#include <string>
#include <vector>

#include "taichi/backends/opengl/opengl_program.h"

namespace taichi {
namespace lang {
namespace opengl {

struct AotData {
  std::unordered_map<std::string, CompiledTaichiKernel> kernels;
  std::unordered_map<std::string, CompiledTaichiKernel> kernel_tmpls;
  std::vector<aot::CompiledFieldData> fields;

  size_t root_buffer_size;

  TI_IO_DEF(kernels, kernel_tmpls, fields, root_buffer_size);
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
