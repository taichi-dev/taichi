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

inline std::string get_project_fn() {
  return fmt::format("{}/projects/taichi_lang/", get_repo_dir());
}

template <typename T>
using Handle = std::shared_ptr<T>;

constexpr int indirect_loop_variable_index = -1;
constexpr int default_simd_width_x86_64 = 8;

class Expr;

enum class Arch { x86_64, gpu };

inline int default_simd_width(Arch arch) {
  if (arch == Arch::x86_64) {
    return default_simd_width_x86_64;
  } else if (arch == Arch::gpu) {
    return 32;
  } else {
    TC_NOT_IMPLEMENTED;
    return -1;
  }
}

struct CompileConfig {
  Arch arch;
  int simd_width;
  int gcc_version;
  bool internal_optimization;
  int external_optimization_level;

  CompileConfig() {
    arch = Arch::x86_64;
    simd_width = default_simd_width(arch);
    internal_optimization = true;
    external_optimization_level = 3;
#if defined(TC_PLATFORM_OSX)
    gcc_version = -1;
#else
    gcc_version = 5;  // not 7 for faster compilation
#endif
  }

  std::string compiler_name() {
    if (gcc_version == -1) {
      return "gcc";
    } else {
      return fmt::format("gcc-{}", gcc_version);
    }
  }

  std::string gcc_opt_flag() {
    TC_ASSERT(0 <= external_optimization_level &&
              external_optimization_level < 5);
    if (external_optimization_level < 4) {
      return fmt::format("-O{}", external_optimization_level);
    } else
      return "-Ofast";
  }

  std::string compile_cmd(const std::string &input,
                          const std::string &output,
                          bool verbose = false) {
    auto cmd = fmt::format(
        "{} {} -std=c++14 -shared -fPIC {} -march=native -I {}/headers "
        "-Wall "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_CPU -o {} -lstdc++",
        compiler_name(), input, gcc_opt_flag(), get_project_fn(), output);

    if (!verbose) {
      cmd += fmt::format(" 2> {}.log", input);
    }
    return cmd;
  }
};

enum class Device { cpu, gpu };

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
  none, // "void"
  unknown
};

template <typename T>
DataType get_data_type() {
  if (std::is_same<T, float32>()) {
    return DataType::f32;
  } else if (std::is_same<T, int32>()) {
    return DataType::i32;
  } else {
    TC_NOT_IMPLEMENTED;
  }
  return DataType::unknown;
}

inline std::string data_type_name(DataType t) {
  static std::map<DataType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_DATA_TYPE(i, j) type_names[DataType::i] = #j;
    REGISTER_DATA_TYPE(f16, float16);
    REGISTER_DATA_TYPE(f32, float32);
    REGISTER_DATA_TYPE(f64, float64);
    REGISTER_DATA_TYPE(i8, int8);
    REGISTER_DATA_TYPE(i16, int16);
    REGISTER_DATA_TYPE(i32, int32);
    REGISTER_DATA_TYPE(i64, int64);
    REGISTER_DATA_TYPE(u8, uint8);
    REGISTER_DATA_TYPE(u16, uint16);
    REGISTER_DATA_TYPE(u32, uint32);
    REGISTER_DATA_TYPE(u64, uint64);
    REGISTER_DATA_TYPE(none, none);
    REGISTER_DATA_TYPE(unknown, unknown);
#undef REGISTER_DATA_TYPE
  }
  return type_names[t];
}

enum class SNodeType {
  undefined,
  fixed,
  dynamic,
  forked,
  place,
  hashed,
  pointer,
  indirect,
};

inline std::string snode_type_name(SNodeType t) {
  static std::map<SNodeType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[SNodeType::i] = #i;
    REGISTER_TYPE(undefined);
    REGISTER_TYPE(fixed);
    REGISTER_TYPE(dynamic);
    REGISTER_TYPE(forked);
    REGISTER_TYPE(place);
    REGISTER_TYPE(hashed);
    REGISTER_TYPE(pointer);
    REGISTER_TYPE(indirect);
#undef REGISTER_TYPE
  }
  return type_names[t];
}

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
  cmp_lt,
  cmp_le,
  cmp_gt,
  cmp_ge,
  cmp_eq,
  cmp_ne,
  undefined
};

inline std::string binary_type_name(BinaryType type) {
  static std::map<BinaryType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[BinaryType::i] = #i;
    REGISTER_TYPE(mul);
    REGISTER_TYPE(add);
    REGISTER_TYPE(sub);
    REGISTER_TYPE(div);
    REGISTER_TYPE(mod);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(cmp_lt);
    REGISTER_TYPE(cmp_le);
    REGISTER_TYPE(cmp_gt);
    REGISTER_TYPE(cmp_ge);
    REGISTER_TYPE(cmp_ne);
    REGISTER_TYPE(cmp_eq);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

inline std::string binary_type_symbol(BinaryType type) {
  static std::map<BinaryType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i, s) type_names[BinaryType::i] = #s;
    REGISTER_TYPE(mul, *);
    REGISTER_TYPE(add, +);
    REGISTER_TYPE(sub, -);
    REGISTER_TYPE(div, /);
    REGISTER_TYPE(mod, %);
    REGISTER_TYPE(max, max);
    REGISTER_TYPE(min, min);
    REGISTER_TYPE(cmp_lt, <);
    REGISTER_TYPE(cmp_le, <=);
    REGISTER_TYPE(cmp_gt, >);
    REGISTER_TYPE(cmp_ge, >=);
    REGISTER_TYPE(cmp_ne, !=);
    REGISTER_TYPE(cmp_eq, ==);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

enum class NodeType : int {
  binary,  // regular binary operators that work for both int and float, and
           // return the same type
  land,
  load,
  store,
  pointer,
  combine,
  index,
  addr,
  adapter_store,
  adapter_load,
  imm,
  floor,
  sqrt,
  inv,
  neg,
  cast,
  shr,
  shl,
  cmp,
  select,
  // vectorized
  vload,
  vload1,
  vstore,
  touch,
  print,
  reduce,
  gather,
  // locals
  alloca,  // allocate on stack (local variable)
  local_load,
  local_store,
  undefined
};

inline std::string node_type_name(NodeType type) {
  static std::map<NodeType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[NodeType::i] = #i;
    REGISTER_TYPE(binary);
    REGISTER_TYPE(land);
    REGISTER_TYPE(load);
    REGISTER_TYPE(store);
    REGISTER_TYPE(combine);
    REGISTER_TYPE(addr);
    REGISTER_TYPE(pointer);
    REGISTER_TYPE(adapter_store);
    REGISTER_TYPE(adapter_load);
    REGISTER_TYPE(imm);
    REGISTER_TYPE(index);
    REGISTER_TYPE(floor);
    REGISTER_TYPE(sqrt);
    REGISTER_TYPE(inv);
    REGISTER_TYPE(neg);
    REGISTER_TYPE(cast);
    REGISTER_TYPE(shr);
    REGISTER_TYPE(shl);
    REGISTER_TYPE(cmp);
    REGISTER_TYPE(vload);
    REGISTER_TYPE(vload1);
    REGISTER_TYPE(vstore);
    REGISTER_TYPE(touch);
    REGISTER_TYPE(select);
    REGISTER_TYPE(print);
    REGISTER_TYPE(reduce);
    REGISTER_TYPE(gather);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

enum class CmpType { eq, ne, le, lt };

constexpr int max_num_indices = 4;

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
void initialize_benchmark();
TC_NAMESPACE_END
