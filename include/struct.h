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

#if defined(TC_GPU)
TC_FORCE_INLINE __device__ void *allocate(std::size_t size) {
  return taichi::Tlang::UnifiedAllocator::alloc(*device_head, size);
}
template <typename T>
TC_FORCE_INLINE __device__ T *allocate() {
  return new (allocate(sizeof(T))) T();
}
#else
TC_FORCE_INLINE __host__ void *allocate(std::size_t size) {
  return taichi::Tlang::allocator()->alloc(size);
}
template <typename T>
TC_FORCE_INLINE __host__ T *allocate() {
  auto addr = taichi::Tlang::allocator()->alloc(sizeof(T));
  return new (addr) T();
}
#endif

using PhysicalIndexGroup = int[max_num_indices];

template <typename T>
struct SNodeID;

struct SNodeMeta {
  bool active;
  int ptr;
  int indices[max_num_indices];
};

template <typename T>
struct SNodeAllocator {
  SNodeMeta *meta_pool;
  using data_type = typename T::child_type;
  static constexpr std::size_t pool_size =
      (1LL << 33) /
      sizeof(data_type);  // each snode allocator takes at most 8 GB
  data_type *data_pool;
  int tail;
  static constexpr int id = SNodeID<T>::value;

  TC_DEVICE SNodeAllocator() {
    if (T::has_null)
      data_pool = (data_type *)allocate(sizeof(data_type) * pool_size);
    else
      data_pool = nullptr;
    meta_pool = (SNodeMeta *)allocate(sizeof(SNodeMeta) * pool_size);
  }

  __device__ __host__ void reset_meta() {
    tail = 0;
  }

#if defined(TC_GPU)
  __device__ data_type *allocate_node(const PhysicalIndexGroup &index) {
    auto id = atomicAdd(&tail, 1);
    SNodeMeta &meta = meta_pool[id];
    meta.active = true;
    meta.ptr = id;

    for (int i = 0; i < max_num_indices; i++)
      meta.indices[i] = index[i];

    return new (data_pool + id) data_type();
  }
#else
  data_type *allocate_node(const PhysicalIndexGroup &index) {
    TC_ASSERT(this != nullptr);
    TC_ASSERT(data_pool != nullptr);
    TC_ASSERT(meta_pool != nullptr);
    auto id = atomicAdd(&tail, 1);
    SNodeMeta &meta = meta_pool[id];
    meta.active = true;
    meta.ptr = id;

    for (int i = 0; i < max_num_indices; i++)
      meta.indices[i] = index[i];

    return new (data_pool + id) data_type();
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

  SNodeManager() {
    allocator = create_unified<Allocator>();
  }

  Allocator *get_allocator() {
    return allocator;
  }
};

struct Managers {
  void *managers[max_num_snodes];

  Managers() {
  }

  template <typename T>
  SNodeManager<T> *&get_manager() {
    return (SNodeManager<T> *&)(managers[SNodeID<T>::value]);
  }

#if defined(TC_STRUCT)
  static void initialize() {
    auto addr = create_unified<Managers>();
    TC_ASSERT(addr == get_instance());
  }
#endif

  template <typename T>
  __host__ __device__ static SNodeManager<T> *&get() {
    return get_instance()->get_manager<T>();
  }

  template <typename T>
  __host__ __device__ static SNodeAllocator<T> *get_allocator() {
    return get<T>()->get_allocator();
  }

  __host__ __device__ static Managers *get_instance() {
#if __CUDA_ARCH__
    return (Managers *)((unsigned char *)(device_data));
#else
    return (Managers *)((unsigned char *)(allocator()->data));
#endif
  }
};

template <typename child_type_>
struct layout_root {
  using child_type = child_type_;
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

  TC_DEVICE TC_FORCE_INLINE void activate(int i,
                                          const PhysicalIndexGroup &index) {
  }

  static constexpr bool has_null = true;
};

template <typename child_type_, int n_>
struct dense {
  using child_type = child_type_;
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

  TC_DEVICE TC_FORCE_INLINE void activate(int i,
                                          const PhysicalIndexGroup &index) {
  }

  static constexpr bool has_null = false;
};

template <typename _child_type>
struct hashed {
  using child_type = _child_type;
  std::unordered_map<int, child_type *> data;
  std::mutex mut;

  hashed() {
    std::cout << "initializing hashed" << std::endl;
  };

  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
    if (data.find(i) == data.end()) {
      return nullptr;
    }
    return data[i];
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i,
                                          const PhysicalIndexGroup &index) {
    if (data.find(i) == data.end()) {
      auto ptr = Managers::get<hashed>()->get_allocator()->allocate_node(index);
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
  int lock;
  // std::mutex mut;

  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
    // TC_ASSERT(i == 0);
    return data;
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return 1;
  }

  TC_DEVICE TC_FORCE_INLINE static int get_max_n() {
    return 1;
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i,
                                          const PhysicalIndexGroup &index) {
#if defined(__CUDA_ARCH__)
    int warp_id = threadIdx.x % 32;
    for (int k = 0; k < 32; k++) {
      if (k == warp_id) {
        while (!atomicCAS(&lock, 0, 1))
          ;
#endif
        if (data == nullptr) {
          data = Managers::get_instance()
                     ->get<pointer>()
                     ->get_allocator()
                     ->allocate_node(index);
        }
#if defined(__CUDA_ARCH__)
        lock = 0;
      }
    }
#endif
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

  TC_DEVICE TC_FORCE_INLINE void activate(int i,
                                          const PhysicalIndexGroup &index) {
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
