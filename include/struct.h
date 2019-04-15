#pragma once
#include "common.h"
#include "arithmetics.h"

// *****************************************************************************
// these structures are used for maintaining metadata and sparsity.
// Their look_up function takes a merged index, but they don't know where do the
// bits come from.

#if defined(TLANG_KERNEL)
#define TC_EXPORT
#if defined(TC_GPU)
#define TC_DEVICE __device__ __host__
#define TLANG_ACCESSOR __device__ __host__ TC_FORCE_INLINE
#else
#define TC_DEVICE
#define TLANG_ACCESSOR TC_FORCE_INLINE
#endif
#else
#define TLANG_ACCESSOR
#undef TC_EXPORT
#define TC_EXPORT extern "C"
#define TC_DEVICE
#endif

TLANG_NAMESPACE_BEGIN

template <typename child_type>
struct layout_root {
  child_type children;
  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
    return &children;
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return 1;
  }

  TC_DEVICE TC_FORCE_INLINE static int get_max_n() {
    return 1;
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i) {
  }

  static constexpr bool has_null = false;
};

template <typename child_type, int n_>
struct dense {
  static constexpr int n = n_;
  child_type children[n];
  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
    return &children[i];
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return n;
  }

  TC_DEVICE TC_FORCE_INLINE static int get_max_n() {
    return n;
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i) {
  }

  static constexpr bool has_null = false;
};

#if defined(TC_GPU)

template <typename T>
TC_FORCE_INLINE __device__ T *allocate() {
  auto addr = taichi::Tlang::UnifiedAllocator::alloc(*device_head, sizeof(T));
  return new (addr) T();
}
#else
template <typename T>
TC_FORCE_INLINE __host__ T *allocate() {
  auto addr = taichi::Tlang::allocator()->alloc(sizeof(T));
  return new (addr) T();
}
#endif

struct SNodeMeta {
  bool active;
  int ptr;
  int indices[max_num_indices];
};

template <typename T>
struct SNodeAllocator {
  static constexpr int pool_size = (1 << 20);
  SNodeMeta *meta_pool;
  T *data_pool;
  int tail;

  TC_DEVICE SNodeAllocator() {
    data_pool = (T *)allocate(sizeof(T) * pool_size);
    meta_pool = (SNodeMeta *)allocate(sizeof(SNodeMeta) * pool_size);
  }

#if defined(TC_GPU)
  __device__ T *allocate_node(int i0, int i1, int i2, int i3) {
    auto id = atomicAdd(&tail, 1);
    SNodeMeta &meta = meta_pool[id];
    meta.active = true;
    meta.ptr = id;

    meta.indices[0] = i0;
    meta.indices[1] = i1;
    meta.indices[2] = i2;
    meta.indices[3] = i3;

    return data_pool[id];
  }
#else
  T *allocate_node(int i0, int i1, int i2, int i3) {
    auto id = atomicAdd(&tail, 1);
    SNodeMeta &meta = meta_pool[id];
    meta.active = true;
    meta.ptr = id;

    meta.indices[0] = i0;
    meta.indices[1] = i1;
    meta.indices[2] = i2;
    meta.indices[3] = i3;

    return data_pool[id];
  }
#endif

  void gc() {
  }

  void print_statistics() {
    std::cout << "  num nodes: " << tail << std::endl;
  }
};

template <typename T>
struct SNodeManager {
  using Allocator = SNodeAllocator<T>;
  Allocator *allocator;

  Allocator &get_allocator() {
    return *allocator;
  }
};

struct Managers {
  void *managers[max_num_snodes];

  Managers() {
  }

  template <typename T>
  SNodeManager<T> *&get(int i) {
    return (SNodeManager<T> *&)(managers[i]);
  }

#if defined(TC_STRUCT)
  static void initialize() {
    auto addr = create_unified<Managers>();
    TC_ASSERT(addr == get_instance());
  }
#endif

  static Managers *get_instance() {
    return (Managers *)((unsigned char *)(*allocator()->head) + sizeof(void *));
  }
};

template <typename _child_type>
struct hashed {
  using child_type = _child_type;
  std::unordered_map<int, child_type *> data;
  std::mutex mut;
  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
#if defined(TLANG_HOST)
    activate(i);
#else
    if (data.find(i) == data.end()) {
      return nullptr;
    }
#endif
    return data[i];
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i) {
    if (data.find(i) == data.end()) {
      child_type *ptr = allocate<child_type>();
      data.insert(std::make_pair(i, ptr));
    }
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return data.size();
  }

  static constexpr bool has_null = true;
};

template <typename _child_type>
struct pointer {
  using child_type = _child_type;
  child_type *data;
  // std::mutex mut;
  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
#if defined(TLANG_HOST)
    activate(i);
#endif
    return data;
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return 1;
  }

  TC_DEVICE TC_FORCE_INLINE static int get_max_n() {
    return 1;
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i) {
    if (data == nullptr) {
      data = allocate<child_type>();
    }
  }

  static constexpr bool has_null = true;
};

template <typename _child_type, int max_n_>
struct dynamic {
  static constexpr int max_n = max_n_;
  using child_type = _child_type;
  child_type data[max_n];
  int n;

  TC_DEVICE dynamic() : n(0) {
  }

  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
#if defined(TLANG_HOST)
    // assuming serial
    n = std::max(n, i + 1);
#endif
    return &data[i];
  }

  __device__ TC_FORCE_INLINE void clear() {
    n = 0;
  }

#if defined(TC_GPU)
  __device__ TC_FORCE_INLINE void append(child_type t) {
    data[atomicAdd(&n, 1)] = t;
  }
#else
  TC_FORCE_INLINE void append(child_type t) {
    data[atomicAdd(&n, 1)] = t;
  }
#endif

  TC_DEVICE TC_FORCE_INLINE void activate(int i) {
    // TC_ASSERT();
    // Do nothing
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return n;
  }

  TC_DEVICE TC_FORCE_INLINE static int get_max_n() {
    return max_n;
  }

  static constexpr bool has_null = false;
};
// *****************************************************************************

template <int max_n_>
struct indirect {
  static constexpr int max_n = max_n_;
  int data[max_n];
  std::atomic<int> n;

  TC_DEVICE indirect() : n(0) {
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() {
    return n;
  }

  TC_DEVICE TC_FORCE_INLINE static int get_max_n() {
    return max_n;
  }

  TC_DEVICE TC_FORCE_INLINE int *look_up(int i) {  // i is flattened index
#if defined(TLANG_HOST)
    n.store(std::max(n.load(), i + 1));
#endif
    return &data[i];
  }

  TC_DEVICE TC_FORCE_INLINE void clear() {
    n.store(0);
  }

  static constexpr bool has_null = false;
};

template <typename T>
struct LeafContext {
  int indices[max_num_indices];
  T *ptr;
};
// *****************************************************************************

TLANG_NAMESPACE_END
