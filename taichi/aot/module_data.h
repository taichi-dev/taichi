#pragma once

#include <string>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"

namespace taichi {
namespace lang {
namespace aot {

struct CompiledFieldData {
  std::string field_name;
  uint32_t dtype{0};
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

}  // namespace aot
}  // namespace lang
}  // namespace taichi
