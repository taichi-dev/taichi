#pragma once

#include "taichi/lang_util.h"
#include "taichi/common/core.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

TLANG_NAMESPACE_BEGIN
namespace cccp {

inline std::string cc_data_type_name(DataType dt) {
  switch (dt) {
    case DataType::i32:
      return "int";
    case DataType::f32:
      return "float";
    case DataType::f64:
      return "double";
    default:
      TI_ERROR("Unsupported DataType={} on C backend", data_type_name(dt));
  }
}
inline std::string cc_type_signature(DataType dt) {
  switch (dt) {
    case DataType::i32:
      return "int";
    case DataType::f32:
      return "float";
    case DataType::f64:
      return "double";
    default:
      TI_ERROR("Unsupported DataType={} on C backend", data_type_name(dt));
  }
}

inline std::string cc_atomic_op_type_symbol(AtomicOpType op) {
  switch (op) {
    case AtomicOpType::add:
      return "+";
    case AtomicOpType::sub:
      return "-";
    case AtomicOpType::bit_or:
      return "|";
    case AtomicOpType::bit_xor:
      return "^";
    case AtomicOpType::bit_and:
      return "&";
    case AtomicOpType::max:
      return "max";
    case AtomicOpType::min:
      return "min";
    default:
      TI_ERROR("Unsupported AtomicOpType={} on C backend", atomic_op_type_name(op));
  }
}

inline bool cc_is_binary_op_infix(BinaryOpType op) {
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

template <typename... Args>
inline int execute(std::string fmt, Args &&... args) {
  auto cmd = fmt::format(fmt, std::forward<Args>(args)...);
  TI_TRACE("Executing command: {}", cmd);
  int ret = std::system(cmd.c_str());
  TI_TRACE("Command exit status: {}", ret);
  return ret;
}

}  // namespace cccp
TLANG_NAMESPACE_END
