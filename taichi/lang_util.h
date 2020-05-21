// Definitions of utility functions and enums
#pragma once

#include "taichi/util/io.h"
#include "taichi/common/core.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

struct Context;

using FunctionType = std::function<void(Context &)>;

enum class DataType : int {
#define PER_TYPE(x) x,
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
};

template <typename T>
inline DataType get_data_type() {
  if (std::is_same<T, float32>()) {
    return DataType::f32;
  } else if (std::is_same<T, float64>()) {
    return DataType::f64;
  } else if (std::is_same<T, bool>()) {
    return DataType::u1;
  } else if (std::is_same<T, int8>()) {
    return DataType::i8;
  } else if (std::is_same<T, int16>()) {
    return DataType::i16;
  } else if (std::is_same<T, int32>()) {
    return DataType::i32;
  } else if (std::is_same<T, int64>()) {
    return DataType::i64;
  } else if (std::is_same<T, uint8>()) {
    return DataType::u8;
  } else if (std::is_same<T, uint16>()) {
    return DataType::u16;
  } else if (std::is_same<T, uint32>()) {
    return DataType::u32;
  } else if (std::is_same<T, uint64>()) {
    return DataType::u64;
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

std::string data_type_name(DataType t);

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

inline bool unary_op_is_cast(UnaryOpType op) {
  return op == UnaryOpType::cast_value || op == UnaryOpType::cast_bits;
}

inline bool constexpr is_trigonometric(UnaryOpType op) {
  return op == UnaryOpType::sin || op == UnaryOpType::asin ||
         op == UnaryOpType::cos || op == UnaryOpType::acos ||
         op == UnaryOpType::tan || op == UnaryOpType::tanh;
}

inline bool constexpr is_real(DataType dt) {
  return dt == DataType::f16 || dt == DataType::f32 || dt == DataType::f64;
}

inline bool constexpr is_integral(DataType dt) {
  return dt == DataType::i8 || dt == DataType::i16 || dt == DataType::i32 ||
         dt == DataType::i64 || dt == DataType::u8 || dt == DataType::u16 ||
         dt == DataType::u32 || dt == DataType::u64;
}

inline bool constexpr is_signed(DataType dt) {
  TI_ASSERT(is_integral(dt));
  return dt == DataType::i8 || dt == DataType::i16 || dt == DataType::i32 ||
         dt == DataType::i64;
}

inline bool constexpr is_unsigned(DataType dt) {
  TI_ASSERT(is_integral(dt));
  return !is_signed(dt);
}

inline bool needs_grad(DataType dt) {
  return is_real(dt);
}

// Regular binary ops:
// Operations that take two oprands, and returns a single operand with the same
// type

enum class BinaryOpType : int {
#define PER_BINARY_OP(x) x,
#include "inc/binary_op.inc.h"
#undef PER_BINARY_OP
};

inline bool binary_is_bitwise(BinaryOpType t) {
  return t == BinaryOpType ::bit_and || t == BinaryOpType ::bit_or ||
         t == BinaryOpType ::bit_xor;
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
  TypedConstant() : dt(DataType::unknown) {
  }

  TypedConstant(DataType dt) : dt(dt) {
    value_bits = 0;
  }

  TypedConstant(int32 x) : dt(DataType::i32), val_i32(x) {
  }

  TypedConstant(float32 x) : dt(DataType::f32), val_f32(x) {
  }

  TypedConstant(int64 x) : dt(DataType::i64), val_i64(x) {
  }

  TypedConstant(float64 x) : dt(DataType::f64), val_f64(x) {
  }

  std::string stringify() const {
    if (dt == DataType::f32) {
      return fmt::format("{}", val_f32);
    } else if (dt == DataType::i32) {
      return fmt::format("{}", val_i32);
    } else if (dt == DataType::i64) {
      return fmt::format("{}", val_i64);
    } else if (dt == DataType::f64) {
      return fmt::format("{}", val_f64);
    } else if (dt == DataType::i8) {
      return fmt::format("{}", val_i8);
    } else if (dt == DataType::i16) {
      return fmt::format("{}", val_i16);
    } else if (dt == DataType::u8) {
      return fmt::format("{}", val_u8);
    } else if (dt == DataType::u16) {
      return fmt::format("{}", val_u16);
    } else if (dt == DataType::u32) {
      return fmt::format("{}", val_u32);
    } else if (dt == DataType::u64) {
      return fmt::format("{}", val_u64);
    } else {
      TI_P(data_type_name(dt));
      TI_NOT_IMPLEMENTED
      return "";
    }
  }

  bool equal_type_and_value(const TypedConstant &o) const {
    if (dt != o.dt)
      return false;
    if (dt == DataType::f32) {
      return val_f32 == o.val_f32;
    } else if (dt == DataType::i32) {
      return val_i32 == o.val_i32;
    } else if (dt == DataType::i64) {
      return val_i64 == o.val_i64;
    } else if (dt == DataType::f64) {
      return val_f64 == o.val_f64;
    } else if (dt == DataType::i8) {
      return val_i8 == o.val_i8;
    } else if (dt == DataType::i16) {
      return val_i16 == o.val_i16;
    } else if (dt == DataType::u8) {
      return val_u8 == o.val_u8;
    } else if (dt == DataType::u16) {
      return val_u16 == o.val_u16;
    } else if (dt == DataType::u32) {
      return val_u32 == o.val_u32;
    } else if (dt == DataType::u64) {
      return val_u64 == o.val_u64;
    } else {
      TI_NOT_IMPLEMENTED
      return false;
    }
  }

  bool operator==(const TypedConstant &o) const {
    return equal_type_and_value(o);
  }

  int32 &val_int32() {
    TI_ASSERT(get_data_type<int32>() == dt);
    return val_i32;
  }

  float32 &val_float32() {
    TI_ASSERT(get_data_type<float32>() == dt);
    return val_f32;
  }

  int64 &val_int64() {
    TI_ASSERT(get_data_type<int64>() == dt);
    return val_i64;
  }

  float64 &val_float64() {
    TI_ASSERT(get_data_type<float64>() == dt);
    return val_f64;
  }
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
