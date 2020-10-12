#pragma once

#include "taichi/lang_util.h"
#include <string>

TLANG_NAMESPACE_BEGIN
namespace opengl {

inline std::string opengl_data_type_name(DataType dt) {
  // https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)
  if (dt == PrimitiveType::f32)
    return "float";
  else if (dt == PrimitiveType::f64)
    return "double";
  else if (dt == PrimitiveType::i32)
    return "int";
  else if (dt == PrimitiveType::i64)
    return "int64_t";
  else
    TI_NOT_IMPLEMENTED;
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
  if (type == PrimitiveType::f32 || type == PrimitiveType::i32)
    return 2;
  else if (type == PrimitiveType::f64 || type == PrimitiveType::i64) {
    return 3;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

inline int opengl_argument_address_shifter(DataType type) {
  return 3 - opengl_data_address_shifter(type);
}

inline int opengl_get_snode_meta_size(const SNode &snode) {
  if (snode.type == SNodeType::dynamic) {
    return sizeof(int);
  } else {
    return 0;
  }
}

}  // namespace opengl
TLANG_NAMESPACE_END
