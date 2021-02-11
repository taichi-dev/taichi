#pragma once

#include "taichi/lang_util.h"
#include "taichi/common/core.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

TLANG_NAMESPACE_BEGIN
namespace cccp {

inline std::string data_type_short_name(DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return "f32";
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return "i32";
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return "i64";
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return "f64";
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return "i8";
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return "i16";
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return "u8";
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return "u16";
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return "u32";
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return "u64";
  } else {
    TI_NOT_IMPLEMENTED
  }
}

inline std::string cc_data_type_name(DataType dt) {
  return "Ti_" + data_type_short_name(dt);
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
      TI_ERROR("Unsupported AtomicOpType={} on C backend",
               atomic_op_type_name(op));
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
inline bool cc_is_unary_op_infix(UnaryOpType op) {
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
