#pragma once

#include <string>

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

enum class MetalDataType : int {
  f32,
  f64,
  i8,
  i16,
  i32,
  i64,
  u8,
  u16,
  u32,
  u64,
  // ptr,
  // none,  // "void"
  unknown
};

MetalDataType to_metal_type(DataType dt);

std::string metal_data_type_name(MetalDataType dt);

inline std::string metal_data_type_name(DataType dt) {
  return metal_data_type_name(to_metal_type(dt));
}

size_t metal_data_type_bytes(MetalDataType dt);

std::string metal_unary_op_type_symbol(UnaryOpType type);

inline std::string metal_binary_op_type_symbol(BinaryOpType type) {
  return binary_op_type_symbol(type);
}

inline bool is_metal_binary_op_infix(BinaryOpType type) {
  return !((type == BinaryOpType::min) || (type == BinaryOpType::max) ||
           (type == BinaryOpType::atan2) || (type == BinaryOpType::pow));
}

}  // namespace metal

TLANG_NAMESPACE_END
