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
  TI_ASSERT(dt->is<PrimitiveType>());
  if (dt->is_primitive(PrimitiveTypeID::i8))
    return "char";
  if (dt->is_primitive(PrimitiveTypeID::i16))
    return "short";
  if (dt->is_primitive(PrimitiveTypeID::i32))
    return "int";
  if (dt->is_primitive(PrimitiveTypeID::i64))
    return "long long";
  if (dt->is_primitive(PrimitiveTypeID::u8))
    return "unsigned char";
  if (dt->is_primitive(PrimitiveTypeID::u16))
    return "unsigned short";
  if (dt->is_primitive(PrimitiveTypeID::u32))
    return "unsigned int";
  if (dt->is_primitive(PrimitiveTypeID::u64))
    return "unsigned long long";
  if (dt->is_primitive(PrimitiveTypeID::f32))
    return "float";
  if (dt->is_primitive(PrimitiveTypeID::f64))
    return "double";
  TI_P(data_type_name(dt));
  TI_NOT_IMPLEMENTED
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
