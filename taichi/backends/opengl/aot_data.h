#pragma once

#include <string>
#include <vector>

#include "taichi/backends/opengl/opengl_program.h"

namespace taichi {
namespace lang {
namespace opengl {

struct CompiledFieldData {
  std::string field_name;
  uint32_t dtype;
  std::string dtype_name;
  size_t mem_offset_in_parent{0};
  std::vector<int> shape;
  bool is_scalar{false};
  int row_num{0};
  int column_num{0};

  TI_IO_DEF(field_name,
            dtype,
            dtype_name,
            mem_offset_in_parent,
            shape,
            is_scalar,
            row_num,
            column_num);
};

struct AotData {
  std::unordered_map<std::string, CompiledTaichiKernel> kernels;
  std::unordered_map<std::string, CompiledTaichiKernel> kernel_tmpls;
  std::vector<CompiledFieldData> fields;

  size_t root_buffer_size;

  TI_IO_DEF(kernels, kernel_tmpls, fields, root_buffer_size);
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
