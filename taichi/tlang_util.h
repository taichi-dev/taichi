// Definitions of utility functions and enums

#pragma once
#include <taichi/arch.h>
#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <taichi/common.h>
#include <taichi/system/profiler.h>

TLANG_NAMESPACE_BEGIN

template <typename T>
using Handle = std::shared_ptr<T>;

constexpr int default_simd_width_x86_64 = 8;

int default_simd_width(Arch arch);

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

struct Context;

using FunctionType = std::function<void(Context &)>;

enum class DataType : int {
  f16,
  f32,
  f64,
  i1,
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
  } else if (std::is_same<T, float64>()) {
    return DataType::f64;
  } else if (std::is_same<T, bool>()) {
    return DataType::i1;
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
    TC_NOT_IMPLEMENTED;
  }
  return DataType::unknown;
}

std::string data_type_name(DataType t);

std::string data_type_short_name(DataType t);

enum class SNodeType {
#define PER_SNODE(x) x,
#include "taichi/inc/snodes.inc.h"
#undef PER_SNODE
};

std::string snode_type_name(SNodeType t);

enum class UnaryOpType : int {
#define PER_UNARY_OP(x) x,
#include "taichi/inc/unary_op.inc.h"
#undef PER_UNARY_OP
};

std::string unary_op_type_name(UnaryOpType type);

inline bool constexpr is_trigonometric(UnaryOpType op) {
  return op == UnaryOpType::sin || op == UnaryOpType::asin ||
         op == UnaryOpType::cos || op == UnaryOpType::acos ||
         op == UnaryOpType::tan || op == UnaryOpType::tanh;
}

inline bool constexpr is_real(DataType dt) {
  return dt == DataType::f16 || dt == DataType::f32 || dt == DataType::f64;
}

inline bool constexpr is_signed(DataType dt) {
  return dt == DataType::i32 || dt == DataType::i64;
}

inline bool constexpr is_unsigned(DataType dt) {
  return dt == DataType::u32 || dt == DataType::u64;
}

inline bool constexpr is_integral(DataType dt) {
  return dt == DataType::i8 || dt == DataType::i16 || dt == DataType::i32 ||
         dt == DataType::i64;
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

enum class AtomicOpType : int { add, sub, max, min };

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
    } else if (dt == DataType::i64) {
      return val_i64 == o.val_i64;
    } else if (dt == DataType::f64) {
      return val_f64 == o.val_f64;
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

  int64 &val_int64() {
    TC_ASSERT(get_data_type<int64>() == dt);
    return val_i64;
  }

  float64 &val_float64() {
    TC_ASSERT(get_data_type<float64>() == dt);
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

int data_type_size(DataType t);
DataType promoted_type(DataType a, DataType b);

struct CompileConfig {
  Arch arch;
  bool debug;
  int simd_width;
  int gcc_version;
  bool internal_optimization;
  bool lazy_compilation;
  bool force_vectorized_global_load;
  bool force_vectorized_global_store;
  int external_optimization_level;
  int max_vector_width;
  bool print_ir;
  bool print_accessor_ir;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool attempt_vectorized_load_cpu;
  bool demote_dense_struct_fors;
  bool use_llvm;
  bool print_struct_llvm_ir;
  bool print_kernel_llvm_ir;
  bool print_kernel_llvm_ir_optimized;
  bool verbose_kernel_launches;
  bool enable_profiler;
  bool verbose;
  bool fast_math;
  DataType default_fp;
  DataType default_ip;
  std::string extra_flags;
  int default_cpu_block_dim;
  int default_gpu_block_dim;

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

extern CompileConfig default_compile_config;
extern std::string compiled_lib_dir;

bool command_exist(const std::string &command);

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
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

TC_NAMESPACE_END
