// Definitions of utility functions and enums
#pragma once

#include "taichi/util/io.h"
#include "taichi/common/core.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/type.h"
#include "taichi/ir/type_factory.h"

TLANG_NAMESPACE_BEGIN

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

struct Context;

using FunctionType = std::function<void(Context &)>;

template <typename T>
inline DataType get_data_type() {
  if (std::is_same<T, float32>()) {
    return PrimitiveType::f32;
  } else if (std::is_same<T, float64>()) {
    return PrimitiveType::f64;
  } else if (std::is_same<T, bool>()) {
    return PrimitiveType::u1;
  } else if (std::is_same<T, int8>()) {
    return PrimitiveType::i8;
  } else if (std::is_same<T, int16>()) {
    return PrimitiveType::i16;
  } else if (std::is_same<T, int32>()) {
    return PrimitiveType::i32;
  } else if (std::is_same<T, int64>()) {
    return PrimitiveType::i64;
  } else if (std::is_same<T, uint8>()) {
    return PrimitiveType::u8;
  } else if (std::is_same<T, uint16>()) {
    return PrimitiveType::u16;
  } else if (std::is_same<T, uint32>()) {
    return PrimitiveType::u32;
  } else if (std::is_same<T, uint64>()) {
    return PrimitiveType::u64;
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

template <typename T>
inline PrimitiveTypeID get_primitive_data_type() {
  if (std::is_same<T, float32>()) {
    return PrimitiveTypeID::f32;
  } else if (std::is_same<T, float64>()) {
    return PrimitiveTypeID::f64;
  } else if (std::is_same<T, bool>()) {
    return PrimitiveTypeID::u1;
  } else if (std::is_same<T, int8>()) {
    return PrimitiveTypeID::i8;
  } else if (std::is_same<T, int16>()) {
    return PrimitiveTypeID::i16;
  } else if (std::is_same<T, int32>()) {
    return PrimitiveTypeID::i32;
  } else if (std::is_same<T, int64>()) {
    return PrimitiveTypeID::i64;
  } else if (std::is_same<T, uint8>()) {
    return PrimitiveTypeID::u8;
  } else if (std::is_same<T, uint16>()) {
    return PrimitiveTypeID::u16;
  } else if (std::is_same<T, uint32>()) {
    return PrimitiveTypeID::u32;
  } else if (std::is_same<T, uint64>()) {
    return PrimitiveTypeID::u64;
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

std::string data_type_name(DataType t);

std::string data_type_format(DataType dt);

std::string data_type_short_name(DataType t);

enum class SNodeType {
#define PER_SNODE(x) x,
#include "taichi/inc/snodes.inc.h"
#undef PER_SNODE
};

std::string snode_type_name(SNodeType t);

bool is_gc_able(SNodeType t);

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

inline bool is_real(DataType dt) {
  return dt->is_primitive(PrimitiveTypeID::f16) ||
         dt->is_primitive(PrimitiveTypeID::f32) ||
         dt->is_primitive(PrimitiveTypeID::f64);
}

inline bool is_integral(DataType dt) {
  return dt->is_primitive(PrimitiveTypeID::i8) ||
         dt->is_primitive(PrimitiveTypeID::i16) ||
         dt->is_primitive(PrimitiveTypeID::i32) ||
         dt->is_primitive(PrimitiveTypeID::i64) ||
         dt->is_primitive(PrimitiveTypeID::u8) ||
         dt->is_primitive(PrimitiveTypeID::u16) ||
         dt->is_primitive(PrimitiveTypeID::u32) ||
         dt->is_primitive(PrimitiveTypeID::u64);
}

inline bool is_signed(DataType dt) {
  TI_ASSERT(is_integral(dt));
  return dt->is_primitive(PrimitiveTypeID::i8) ||
         dt->is_primitive(PrimitiveTypeID::i16) ||
         dt->is_primitive(PrimitiveTypeID::i32) ||
         dt->is_primitive(PrimitiveTypeID::i64);
}

inline bool is_unsigned(DataType dt) {
  TI_ASSERT(is_integral(dt));
  return !is_signed(dt);
}

inline DataType to_unsigned(DataType dt) {
  TI_ASSERT(is_signed(dt));
  if (dt->is_primitive(PrimitiveTypeID::i8))
    return PrimitiveType::u8;
  else if (dt->is_primitive(PrimitiveTypeID::i16))
    return PrimitiveType::u16;
  else if (dt->is_primitive(PrimitiveTypeID::i32))
    return PrimitiveType::u32;
  else if (dt->is_primitive(PrimitiveTypeID::i64))
    return PrimitiveType::u64;
  else
    return PrimitiveType::unknown;
}

inline bool needs_grad(DataType dt) {
  return is_real(dt);
}

// Regular binary ops:
// Operations that take two operands, and returns a single operand with the
// same type

enum class BinaryOpType : int {
#define PER_BINARY_OP(x) x,
#include "inc/binary_op.inc.h"
#undef PER_BINARY_OP
};

inline bool binary_is_bitwise(BinaryOpType t) {
  return t == BinaryOpType ::bit_and || t == BinaryOpType ::bit_or ||
         t == BinaryOpType ::bit_xor || t == BinaryOpType ::bit_shl ||
         t == BinaryOpType ::bit_sar;
}

std::string binary_op_type_name(BinaryOpType type);

inline bool is_comparison(BinaryOpType type) {
  return starts_with(binary_op_type_name(type), "cmp");
}

inline bool is_bit_op(BinaryOpType type) {
  return starts_with(binary_op_type_name(type), "bit");
}

std::string binary_op_type_symbol(BinaryOpType type);

enum class TernaryOpType : int { select, undefined };

std::string ternary_type_name(TernaryOpType type);

enum class AtomicOpType : int { add, sub, max, min, bit_and, bit_or, bit_xor };

std::string atomic_op_type_name(AtomicOpType type);
BinaryOpType atomic_to_binary_op_type(AtomicOpType type);

enum class SNodeOpType : int {
  is_active,
  length,
  activate,
  deactivate,
  append,
  clear,
  undefined
};

std::string snode_op_type_name(SNodeOpType type);

class IRModified {};

class TypedConstant {
 public:
  DataType dt;
  union {
    uint64 value_bits;
    int32 val_i32;
    float32 val_f32;
    int64 val_i64;
    float64 val_f64;
    int8 val_i8;
    int16 val_i16;
    uint8 val_u8;
    uint16 val_u16;
    uint32 val_u32;
    uint64 val_u64;
  };

 public:
  TypedConstant() : dt(PrimitiveType::unknown) {
  }

  TypedConstant(DataType dt) : dt(dt) {
    value_bits = 0;
  }

  TypedConstant(int32 x) : dt(PrimitiveType::i32), val_i32(x) {
  }

  TypedConstant(float32 x) : dt(PrimitiveType::f32), val_f32(x) {
  }

  TypedConstant(int64 x) : dt(PrimitiveType::i64), val_i64(x) {
  }

  TypedConstant(float64 x) : dt(PrimitiveType::f64), val_f64(x) {
  }

  template <typename T>
  TypedConstant(DataType dt, const T &value) : dt(dt) {
    // TODO: loud failure on pointers
    dt.set_is_pointer(false);
    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      val_f32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
      val_i32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
      val_i64 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
      val_f64 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
      val_i8 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
      val_i16 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
      val_u8 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
      val_u16 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
      val_u32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
      val_u64 = value;
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  template <typename T>
  bool equal_value(const T &value) const {
    return equal_type_and_value(TypedConstant(dt, value));
  }

  std::string stringify() const;

  bool equal_type_and_value(const TypedConstant &o) const;

  bool operator==(const TypedConstant &o) const {
    return equal_type_and_value(o);
  }

  int32 &val_int32();
  float32 &val_float32();
  int64 &val_int64();
  float64 &val_float64();
  int8 &val_int8();
  int16 &val_int16();
  uint8 &val_uint8();
  uint16 &val_uint16();
  uint32 &val_uint32();
  uint64 &val_uint64();
  int64 val_int() const;
  uint64 val_uint() const;
  float64 val_float() const;
  float64 val_cast_to_float64() const;
};

inline std::string make_list(const std::vector<std::string> &data,
                             std::string bracket = "") {
  std::string ret = bracket;
  for (int i = 0; i < (int)data.size(); i++) {
    ret += data[i];
    if (i + 1 < (int)data.size()) {
      ret += ", ";
    }
  }
  if (bracket == "<") {
    ret += ">";
  } else if (bracket == "{") {
    ret += "}";
  } else if (bracket == "[") {
    ret += "]";
  } else if (bracket == "(") {
    ret += ")";
  } else if (bracket != "") {
    TI_P(bracket);
    TI_NOT_IMPLEMENTED
  }
  return ret;
}

template <typename T>
std::string make_list(const std::vector<T> &data,
                      std::function<std::string(const T &t)> func,
                      std::string bracket = "") {
  std::vector<std::string> ret(data.size());
  for (int i = 0; i < (int)data.size(); i++) {
    ret[i] = func(data[i]);
  }
  return make_list(ret, bracket);
}

int data_type_size(DataType t);
DataType promoted_type(DataType a, DataType b);

extern std::string compiled_lib_dir;
extern std::string runtime_tmp_dir;

bool command_exist(const std::string &command);

TLANG_NAMESPACE_END

TI_NAMESPACE_BEGIN
void initialize_benchmark();

template <typename T, typename... Args, typename FP = T (*)(Args...)>
FP function_pointer_helper(std::function<T(Args...)>) {
  return nullptr;
}

template <typename T, typename... Args, typename FP = T (*)(Args...)>
FP function_pointer_helper(T (*)(Args...)) {
  return nullptr;
}

template <typename T>
using function_pointer_type =
    decltype(function_pointer_helper(std::declval<T>()));

TI_NAMESPACE_END
