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

inline bool is_opengl_binary_op_infix(BinaryOpType type)
{
  return !((type == BinaryOpType::min) || (type == BinaryOpType::max) ||
           (type == BinaryOpType::atan2) || (type == BinaryOpType::pow));
}

}  // namespace opengl
TLANG_NAMESPACE_END
