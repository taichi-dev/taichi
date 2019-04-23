#pragma once
#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include "../include/common.h"

TLANG_NAMESPACE_BEGIN

std::string get_project_fn();

template <typename T>
using Handle = std::shared_ptr<T>;

constexpr int default_simd_width_x86_64 = 8;

enum class Arch { x86_64, gpu };

int default_simd_width(Arch arch);

struct CompileConfig {
  Arch arch;
  bool debug;
  int simd_width;
  int gcc_version;
  bool internal_optimization;
  bool force_vectorized_global_load;
  bool force_vectorized_global_store;
  int external_optimization_level;
  int max_vector_width;
  int print_ir;
  bool serial_schedule;
  std::string extra_flags;

  CompileConfig();

  std::string compiler_name();

  std::string gcc_opt_flag();

  std::string compiler_config();

  std::string preprocess_cmd(const std::string &input,
                             const std::string &output,
                             const std::string &extra_flags,
                             bool verbose = false);

  std::string compile_cmd(const std::string &input,
                          const std::string &output,
                          const std::string &extra_flags,
                          bool verbose = false);
};

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

struct Context;

using FunctionType = void (*)(Context);

enum class DataType : int {
  f16,
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
  ptr,
  none,  // "void"
  unknown
};

template <typename T>
inline DataType get_data_type() {
  if (std::is_same<T, float32>()) {
    return DataType::f32;
  } else if (std::is_same<T, int32>()) {
    return DataType::i32;
  } else {
    TC_NOT_IMPLEMENTED;
  }
  return DataType::unknown;
}

std::string data_type_name(DataType t);

std::string data_type_short_name(DataType t);

enum class SNodeType {
  undefined,
  root,
  dense,
  dynamic,
  place,
  hashed,
  pointer,
  indirect,
};

std::string snode_type_name(SNodeType t);

enum class UnaryType : int {
  neg,
  sqrt,
  floor,
  cast,
  abs,
  sin,
  cos,
  inv,
  rcp,
  exp,
  log,
  rsqrt,
  bit_not,
  undefined
};

std::string unary_type_name(UnaryType type);

// Regular binary ops:
// Operations that take two oprands, and returns a single operand with the same
// type

enum class BinaryType : int {
  mul,
  add,
  sub,
  div,
  mod,
  max,
  min,
  bit_and,
  bit_or,
  bit_xor,
  cmp_lt,
  cmp_le,
  cmp_gt,
  cmp_ge,
  cmp_eq,
  cmp_ne,
  undefined
};

std::string binary_type_name(BinaryType type);

inline bool is_comparison(BinaryType type) {
  return starts_with(binary_type_name(type), "cmp");
}

std::string binary_type_symbol(BinaryType type);

enum class TrinaryType : int { select, undefined };

std::string trinary_type_name(TrinaryType type);

enum class AtomicType : int { add, max, min };

std::string atomic_type_name(AtomicType type);

enum class SNodeOpType : int { probe, activate, deactivate, append, clear };

std::string snode_op_type_name(SNodeOpType type);

class IRModifiedException {};

class TypedConstant {
 public:
  DataType dt;
  union {
    uint64 value_bits;
    int32 val_i32;
    float32 val_f32;
  };

 public:
  TypedConstant() : dt(DataType::unknown) {
  }

  TypedConstant(int32 x) : dt(DataType::i32), val_i32(x) {
  }

  TypedConstant(float32 x) : dt(DataType::f32), val_f32(x) {
  }

  std::string stringify() const {
    if (dt == DataType::f32) {
      return fmt::format("{}", val_f32);
    } else if (dt == DataType::i32) {
      return fmt::format("{}", val_i32);
    } else {
      TC_P(data_type_name(dt));
      TC_NOT_IMPLEMENTED
      return "";
    }
  }

  bool equal_type_and_value(const TypedConstant &o) {
    if (dt != o.dt)
      return false;
    if (dt == DataType::f32) {
      return val_f32 == o.val_f32;
    } else if (dt == DataType::i32) {
      return val_i32 == o.val_i32;
    } else {
      TC_NOT_IMPLEMENTED
      return false;
    }
  }

  int32 &val_int32() {
    TC_ASSERT(get_data_type<int32>() == dt);
    return val_i32;
  }

  float32 &val_float32() {
    TC_ASSERT(get_data_type<float32>() == dt);
    return val_f32;
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
    TC_P(bracket);
    TC_NOT_IMPLEMENTED
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

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
void initialize_benchmark();
TC_NAMESPACE_END
