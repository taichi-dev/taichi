#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include <string>

TLANG_NAMESPACE_BEGIN

namespace opengl {

inline std::string opengl_data_type_name(DataType dt) {
  // https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)
  dt.set_is_pointer(false);
  if (dt->is_primitive(PrimitiveTypeID::f32))
    return "float";
  else if (dt->is_primitive(PrimitiveTypeID::f64))
    return "double";
  else if (dt->is_primitive(PrimitiveTypeID::i32))
    return "int";
  else if (dt->is_primitive(PrimitiveTypeID::i64))
    return "int64_t";
  else if (dt->is_primitive(PrimitiveTypeID::u32))
    return "uint";
  else if (dt->is_primitive(PrimitiveTypeID::u64))
    return "uint64_t";
  else {
    TI_ERROR("Type {} not supported.", dt->to_string());
  }
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
  // TODO: fail loudly when feeding a pointer type to this function, after type
  // system upgrade.
  type.set_is_pointer(false);
  auto dtype_size = data_type_size(type);
  if (dtype_size == 4) {
    return 2;
  } else if (dtype_size == 8) {
    return 3;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

inline int opengl_argument_address_shifter(DataType type) {
  return 3 - opengl_data_address_shifter(type);
}

}  // namespace opengl
TLANG_NAMESPACE_END
