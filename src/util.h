#pragma once
#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <immintrin.h>
#include "../headers/common.h"

#define TLANG_NAMESPACE_BEGIN \
  namespace taichi {          \
  namespace Tlang {
#define TLANG_NAMESPACE_END \
  }                         \
  }

TLANG_NAMESPACE_BEGIN

std::string get_project_fn();

template <typename T>
using Handle = std::shared_ptr<T>;

constexpr int default_simd_width_x86_64 = 8;

enum class Arch { x86_64, gpu };

int default_simd_width(Arch arch);

enum class Device {
  cpu,
  gpu
};

struct CompileConfig {
  Arch arch;
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

  std::string compile_cmd(const std::string &input,
                          const std::string &output,
                          bool verbose = false);
};

class AlignedAllocator {
  std::vector<uint8> _data;
  void *data;
  void *_cuda_data;
  std::size_t size;

 public:
  Device device;

  AlignedAllocator() {
    data = nullptr;
  }

  AlignedAllocator(std::size_t size, Device device = Device::cpu);

  ~AlignedAllocator();

  void memset(unsigned char val) {
    std::memset(data, val, size);
  }

  bool initialized() const {
    return data != nullptr;
  }

  template <typename T = void>
  T *get() {
    TC_ASSERT(initialized());
    return reinterpret_cast<T *>(data);
  }

  AlignedAllocator operator=(const AlignedAllocator &) = delete;

  AlignedAllocator(AlignedAllocator &&o) noexcept {
    (*this) = std::move(o);
  }

  AlignedAllocator &operator=(AlignedAllocator &&o) noexcept {
    std::swap(_data, o._data);
    data = o.data;
    o.data = nullptr;
    device = o.device;
    size = o.size;
    _cuda_data = o._cuda_data;
    return *this;
  }
};

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

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

template<typename T>
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
  fixed,
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

enum class CmpType { eq, ne, le, lt };

constexpr int max_num_indices = 4;

class IRModifiedException {};

class TypedConstant {
 public:
  DataType dt;

 public:
  union {
    uint64 value_bits;
    int32 val_i32;
    float32 val_f32;
  };
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
};

template <typename T>
std::string make_list(const std::vector<T> &data,
                      std::function<std::string(const T &t)> func,
                      std::string bracket = "") {
  std::string ret = bracket;
  for (int i = 0; i < (int)data.size(); i++) {
    ret += func(data[i]);
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

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
void initialize_benchmark();
TC_NAMESPACE_END
