#pragma once

#include <string>
#include <vector>

#include "taichi/backends/opengl/opengl_program.h"

namespace taichi {
namespace lang {
namespace opengl {

struct AotCompiledKernel {
  CompiledProgram program;
  std::string identifier;

  TI_IO_DEF(program, identifier);
};

struct AotCompiledKernelTmpl {
  std::unordered_map<std::string, CompiledProgram> program;
  std::string identifier;

  TI_IO_DEF(program, identifier);
};

struct CompiledFieldData {
  std::string field_name;
  uint32_t dtype;
  std::string dtype_name;
  std::vector<int> shape;
  bool is_scalar{false};
  int row_num{0};
  int column_num{0};

  TI_IO_DEF(field_name,
            dtype,
            dtype_name,
            shape,
            is_scalar,
            row_num,
            column_num);
};

struct AotData {
  std::vector<AotCompiledKernel> kernels;
  std::vector<AotCompiledKernelTmpl> kernel_tmpls;
  std::vector<CompiledFieldData> fields;

  size_t root_buffer_size;

  TI_IO_DEF(kernels, kernel_tmpls, fields, root_buffer_size);
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
