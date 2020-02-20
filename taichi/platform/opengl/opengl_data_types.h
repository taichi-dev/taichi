#pragma once

#include <taichi/tlang_util.h>
#include <string>

TLANG_NAMESPACE_BEGIN
namespace opengl {

inline std::string opengl_data_type_name(DataType dt)
{
  // https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)
  switch (dt) {
    case DataType::f32:
      return "float";
    case DataType::f64:
      return "double";
    case DataType::i32:
      return "int";
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return "";
}

inline std::string opengl_unary_op_type_symbol(UnaryOpType type)
{
  switch (type)
  {
  case UnaryOpType::neg:
    return "-";
  case UnaryOpType::sqrt:
    return "sqrt";
  case UnaryOpType::floor:
    return "floor";
  case UnaryOpType::ceil:
    return "ceil";
  case UnaryOpType::abs:
    return "abs";
  case UnaryOpType::sgn:
    return "sign";
  case UnaryOpType::sin:
    return "sin";
  case UnaryOpType::asin:
    return "asin";
  case UnaryOpType::cos:
    return "cos";
  case UnaryOpType::acos:
    return "acos";
  case UnaryOpType::tan:
    return "tan";
  case UnaryOpType::tanh:
    return "tanh";
  case UnaryOpType::exp:
    return "exp";
  case UnaryOpType::log:
    return "log";
  default:
    TI_NOT_IMPLEMENTED;
  }
  return "";
}

inline bool is_opengl_binary_op_infix(BinaryOpType type)
{
  return !((type == BinaryOpType::min) || (type == BinaryOpType::max) ||
           (type == BinaryOpType::atan2) || (type == BinaryOpType::pow));
}

}  // namespace opengl
TLANG_NAMESPACE_END
