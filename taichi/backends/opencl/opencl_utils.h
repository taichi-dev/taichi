#pragma once

#include "taichi/lang_util.h"
#include "taichi/common/core.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

TLANG_NAMESPACE_BEGIN
namespace opencl {

inline std::string opencl_data_type_name(DataType dt) {
  return "Ti_" + data_type_short_name(dt);
}

inline bool opencl_is_binary_op_infix(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::max:
    case BinaryOpType::min:
    case BinaryOpType::atan2:
    case BinaryOpType::pow:
      return false;
    default:
      return true;
  }
}
inline bool opencl_is_unary_op_infix(UnaryOpType op) {
  switch (op) {
    case UnaryOpType::neg:
    case UnaryOpType::bit_not:
    case UnaryOpType::logic_not:
      return true;
    default:
      return false;
  }
}

// TODO: move this to lang_util.h:
inline std::string unary_op_type_symbol(UnaryOpType op) {
  switch (op) {
    case UnaryOpType::neg:
      return "-";
    case UnaryOpType::bit_not:
      return "~";
    case UnaryOpType::logic_not:
      return "!";
    default:
      return unary_op_type_name(op);
  }
}

}  // namespace opencl
TLANG_NAMESPACE_END
