#pragma once
#include <taichi/common/util.h>
#include <immintrin.h>
#include "../headers/common.h"

#define TLANG_NAMESPACE_BEGIN \
  namespace taichi {          \
  namespace Tlang {
#define TLANG_NAMESPACE_END \
  }                         \
  }

TLANG_NAMESPACE_BEGIN

enum class Arch { x86_64, gpu };

inline int default_simd_width(Arch arch) {
  if (arch == Arch::x86_64) {
    return 8;  // AVX2
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

  CompileConfig() {
    arch = Arch::x86_64;
    simd_width = -1;
    gcc_version = 5;
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
  static std::map<DataType, std::string> data_type_names;
  if (data_type_names.empty()) {
#define REGISTER_DATA_TYPE(i, j) data_type_names[DataType::i] = #j;
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
#undef REGISTER_DATA_TYPE
  }
  return data_type_names[t];
}

enum class SNodeType {
  undefined,
  fixed,
  dynamic,
  forked,
  place,
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
    REGISTER_TYPE(indirect);
#undef REGISTER_TYPE
  }
  return type_names[t];
}

// Regular binary ops:
// Operations that take two oprands, and returns a single operand with the same
// type

enum class BinaryType : int { mul, add, sub, div, mod, max, min, undefined };

inline std::string binary_type_name(BinaryType type) {
  static std::map<BinaryType, std::string> binary_type_names;
  if (binary_type_names.empty()) {
#define REGISTER_BINARY_TYPE(i) binary_type_names[BinaryType::i] = #i;
    REGISTER_BINARY_TYPE(mul);
    REGISTER_BINARY_TYPE(add);
    REGISTER_BINARY_TYPE(sub);
    REGISTER_BINARY_TYPE(div);
    REGISTER_BINARY_TYPE(mod);
    REGISTER_BINARY_TYPE(max);
    REGISTER_BINARY_TYPE(min);
  }
  return binary_type_names[type];
}

enum class NodeType : int {
  binary,  // regular binary
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
  cast,
  shr,
  shl,
  cmp,
  select,
  // vectorized
  vload,
  vstore,
  undefined
};

inline std::string node_type_name(NodeType type) {
  static std::map<NodeType, std::string> node_type_names;
  if (node_type_names.empty()) {
#define REGISTER_NODE_TYPE(i) node_type_names[NodeType::i] = #i;
    REGISTER_NODE_TYPE(binary);
    REGISTER_NODE_TYPE(land);
    REGISTER_NODE_TYPE(load);
    REGISTER_NODE_TYPE(store);
    REGISTER_NODE_TYPE(combine);
    REGISTER_NODE_TYPE(addr);
    REGISTER_NODE_TYPE(pointer);
    REGISTER_NODE_TYPE(adapter_store);
    REGISTER_NODE_TYPE(adapter_load);
    REGISTER_NODE_TYPE(imm);
    REGISTER_NODE_TYPE(index);
    REGISTER_NODE_TYPE(floor);
    REGISTER_NODE_TYPE(cast);
    REGISTER_NODE_TYPE(shr);
    REGISTER_NODE_TYPE(shl);
    REGISTER_NODE_TYPE(cmp);
    REGISTER_NODE_TYPE(vload);
    REGISTER_NODE_TYPE(vstore);
    REGISTER_NODE_TYPE(select);
  }
  return node_type_names[type];
}

enum class CmpType { eq, ne, le, lt };

constexpr int max_num_indices = 4;

TLANG_NAMESPACE_END
