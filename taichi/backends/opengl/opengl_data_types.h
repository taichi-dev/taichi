#pragma once

#include "taichi/lang_util.h"
#include <string>

TLANG_NAMESPACE_BEGIN
namespace opengl {

inline std::string opengl_data_type_name(DataType dt) {
  // https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)
  switch (dt) {
    case DataType::f32:
      return "float";
    case DataType::f64:
      return "double";
    case DataType::i32:
      return "int";
    case DataType::i64:
      return "int64_t";
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return "";
}

inline bool is_opengl_binary_op_infix(BinaryOpType type) {
  return !((type == BinaryOpType::min) || (type == BinaryOpType::max) ||
           (type == BinaryOpType::atan2) || (type == BinaryOpType::pow));
}

inline bool is_opengl_binary_op_different_return_type(BinaryOpType type) {
  return (type == BinaryOpType::cmp_ne) || (type == BinaryOpType::cmp_eq) ||
         (type == BinaryOpType::cmp_lt) || (type == BinaryOpType::cmp_gt) ||
         (type == BinaryOpType::cmp_le) || (type == BinaryOpType::cmp_ge);
}

inline int opengl_data_address_shifter(DataType type) {
  switch (type) {
    case DataType::f32:
    case DataType::i32:
      return 2;
    case DataType::f64:
    case DataType::i64:
      return 3;
    default:
      TI_NOT_IMPLEMENTED
  }
}

inline int opengl_argument_address_shifter(DataType type) {
  return 3 - opengl_data_address_shifter(type);
}

}  // namespace opengl
TLANG_NAMESPACE_END
