#pragma once

#include <string>

#include "taichi/common/core.h"

namespace taichi {
namespace lang {

enum class UnaryOpType : int {
#define PER_UNARY_OP(x) x,
#include "taichi/inc/unary_op.inc.h"
#undef PER_UNARY_OP
};

std::string unary_op_type_name(UnaryOpType type);

inline bool constexpr unary_op_is_cast(UnaryOpType op) {
  return op == UnaryOpType::cast_value || op == UnaryOpType::cast_bits;
}

inline bool constexpr is_trigonometric(UnaryOpType op) {
  return op == UnaryOpType::sin || op == UnaryOpType::asin ||
         op == UnaryOpType::cos || op == UnaryOpType::acos ||
         op == UnaryOpType::tan || op == UnaryOpType::tanh;
}

// Regular binary ops:
// Operations that take two operands, and returns a single operand with the
// same type

enum class BinaryOpType : int {
#define PER_BINARY_OP(x) x,
#include "taichi/inc/binary_op.inc.h"
#undef PER_BINARY_OP
};

inline bool binary_is_bitwise(BinaryOpType t) {
  return t == BinaryOpType ::bit_and || t == BinaryOpType ::bit_or ||
         t == BinaryOpType ::bit_xor || t == BinaryOpType ::bit_shl ||
         t == BinaryOpType ::bit_sar;
}

inline bool binary_is_logical(BinaryOpType t) {
  return t == BinaryOpType ::logical_and || t == BinaryOpType ::logical_or;
}

std::string binary_op_type_name(BinaryOpType type);

inline bool is_shift_op(BinaryOpType type) {
  return type == BinaryOpType::bit_sar || type == BinaryOpType::bit_shl ||
         type == BinaryOpType::bit_shr;
}

inline bool is_comparison(BinaryOpType type) {
  return starts_with(binary_op_type_name(type), "cmp");
}

inline bool is_bit_op(BinaryOpType type) {
  return starts_with(binary_op_type_name(type), "bit");
}

std::string binary_op_type_symbol(BinaryOpType type);

enum class TernaryOpType : int { select, ifte, undefined };

std::string ternary_type_name(TernaryOpType type);

enum class AtomicOpType : int { add, sub, max, min, bit_and, bit_or, bit_xor };

std::string atomic_op_type_name(AtomicOpType type);
BinaryOpType atomic_to_binary_op_type(AtomicOpType type);

enum class SNodeOpType : int {
  is_active,
  length,
  get_addr,
  activate,
  deactivate,
  append,
  clear,
  undefined
};

std::string snode_op_type_name(SNodeOpType type);

enum class TextureOpType : int { sample_lod, fetch_texel, undefined };

std::string texture_op_type_name(TextureOpType type);

}  // namespace lang
}  // namespace taichi
