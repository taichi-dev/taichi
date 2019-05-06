#pragma once

#include "common.h"
#include "arithmetics.h"
#if defined(TLANG_GPU)
#include <cuda_runtime.h>
#endif

// *****************************************************************************
// these structures are used for maintaining metadata and sparsity.
// Their look_up function takes a merged index, but they don't know where do the
// bits come from.

#if defined(TLANG_KERNEL)
#define TC_EXPORT
#if defined(TLANG_GPU)
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

using PhysicalIndexGroup = int[max_num_indices];

template <typename T>
struct SNodeID;

template <typename T>
__host__ __device__ void get_corner_coord(const PhysicalIndexGroup &indices,
                                          PhysicalIndexGroup &corner);

/*
static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long), "");
static_assert(sizeof(std::uint64_t) == sizeof(unsigned long), "");
// static_assert(std::is_same_v<std::uint64_t, unsigned long long>, "");
static_assert(std::is_same_v<std::uint64_t, unsigned long int>, "");
*/

template <typename T>
struct SNodeAllocator {
  using data_type = typename T::child_type;
  static constexpr std::size_t pool_size = std::min(
      (1ULL << 33) / sizeof(data_type),
      1ULL << 25);  // each snode allocator takes at most 8 GB, max 32M metas
  static constexpr int id = SNodeID<T>::value;

  SNodeMeta *resident_pool;
  SNodeMeta *recycle_pool;
  data_type *data_pool;
  size_t resident_tail;
  size_t recycle_tail;
  size_t execution_tail;

  // backups that will not change during a single kernel execution
  size_t resident_tail_const;
  size_t recycle_tail_const;

  SNodeAllocator() {
    if (T::has_null)
      data_pool = (data_type *)allocate(sizeof(data_type) * pool_size,
                                        sizeof(data_type));
    else
      data_pool = nullptr;
    resident_pool =
        (SNodeMeta *)allocate(sizeof(SNodeMeta) * pool_size, sizeof(SNodeMeta));
    recycle_pool =
        (SNodeMeta *)allocate(sizeof(SNodeMeta) * pool_size, sizeof(SNodeMeta));

    resident_tail = 0;
    recycle_tail = 0;
  }

  __device__ __host__ void reset_meta() {
    resident_tail = 0;
    recycle_tail = 0;
  }

  __host__ __device__ SNodeMeta *allocate_node(
      const PhysicalIndexGroup &index) {
    TC_ASSERT(this != nullptr);
    TC_ASSERT(data_pool != nullptr);
    TC_ASSERT(resident_pool != nullptr);
    auto id = atomic_add(&resident_tail, 1UL);
    TC_ASSERT(id < pool_size);
    SNodeMeta &meta = resident_pool[id];
    meta.active = true;
    meta.ptr = data_pool + id;

    PhysicalIndexGroup corner;
    get_corner_coord<T>(index, corner);

    for (int i = 0; i < max_num_indices; i++) {
      meta.indices[i] = corner[i];
    }

    new (meta.ptr) data_type();
    return &meta;
  }

  void gc() {
  }

  static_assert(sizeof(data_type) % 4 == 0, "");

  __host__ void clear(int flags);

  __host__ AllocatorStat get_stat() {
#if defined(TLANG_GPU)
    cudaDeviceSynchronize();
#endif
    AllocatorStat stat;
    stat.snode_id = SNodeID<T>::value;
    stat.pool_size = pool_size;
    stat.num_recycled_blocks = recycle_tail;
    stat.num_resident_blocks = resident_tail;
    stat.resident_metas = resident_pool;
    return stat;
  }
};

template <typename T>
struct SNodeManager {
  using Allocator = SNodeAllocator<T>;
  Allocator *allocator;

  SNodeManager() {
    allocator = create_unified<Allocator>();
  }

  __host__ __device__ Allocator *get_allocator() {
    return allocator;
  }
};

struct Managers {
  void *managers[max_num_snodes];

  Managers() {
  }

  template <typename T>
  __host__ __device__ SNodeManager<T> *&get_manager() {
    return (SNodeManager<T> *&)(managers[SNodeID<T>::value]);
  }

  __host__ __device__ static void initialize() {
    auto addr = create_unified<Managers>();
    TC_ASSERT(addr == get_instance());
  }

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

#if defined(TLANG_GPU)

inline constexpr std::size_t least_pot_bound(std::size_t v) {
  std::size_t ret = 1;
  while (ret < v) {
    ret *= 2;
  }
  return ret;
}

static_assert(sizeof(unsigned long long) == sizeof(unsigned long), "");

template <typename T>
__global__ void recycle_all_gpu(SNodeAllocator<T> *allocator, int flags) {
  auto num_blocks = allocator->resident_tail;
  for (int b = blockIdx.x; b < num_blocks; b += gridDim.x) {
    auto t = threadIdx.x;
    /*
    if (allocator->resident_pool[b].active)
      return;  // still active, do nothing
    */
    // zero-fill
    auto &meta = allocator->resident_pool[b];
    if (t == 0 && flags)
      *(meta.snode_ptr) = nullptr;
    auto ptr = (int *)(meta.ptr);
    while (t * sizeof(int) < sizeof(T::child_type)) {
      ptr[t] = 0;
      t += blockDim.x;
    }

    /*
    // push to recycle list
    if (t == 0) {
      auto x = atomic_add(&allocator->recycle_tail, 1);
      allocator->recycle_pool[x] = allocator->resident_pool[b];
    }
    */
  }
}

template <typename T>
__global__ void reset_execution_tail() {
  Managers::get_allocator<T>()->execution_tail = 0;
}

template <typename T>
__global__ void reset_tails() {
  Managers::get_allocator<T>()->resident_tail = 0;
  Managers::get_allocator<T>()->recycle_tail = 0;
}

template <typename T>
__global__ void backup_tails() {
  auto allocator = Managers::get_allocator<T>();
  allocator->resident_tail_const = allocator->resident_tail;
  allocator->recycle_tail_const = allocator->recycle_tail;
}

template <typename T>
__host__ void SNodeAllocator<T>::clear(int flags) {
  int blockDim = 256;  // least_pot_bound(sizeof(data_type) / sizeof(int));
#if defined(TL_DEBUG)
  cudaDeviceSynchronize();
  printf("tail    %d size %d blockDim %d\n", resident_tail, sizeof(data_type),
         blockDim);
#endif
  gpu_runtime_init();
#if defined(TL_DEBUG)
  printf("gc ");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif
  recycle_all_gpu<<<2048, blockDim>>>(this, flags);
#if defined(TL_DEBUG)
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "device  " << milliseconds << " ms" << std::endl;
#endif

#if defined(TL_DEBUG)
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err) {
    printf("CUDA Error (File %s Ln %d): %s\n", __FILE__, __LINE__,
           cudaGetErrorString(err));
    exit(-1);
  }
#endif
  if (flags) {
    reset_tails<T><<<1, 1>>>();
  }
}
#else
template <typename T>
void SNodeAllocator<T>::clear(int flags) {
  printf("not implemented\n");
  exit(-1);
}
#endif

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

/*
template <typename _child_type>
struct ghashed {
  using child_type = _child_type;
  int n;
  int lock;


  ghashed() {
  };

  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {
  }

  TC_DEVICE TC_FORCE_INLINE void activate(int i,
                                          const PhysicalIndexGroup &index) {
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return n;
  }

  static constexpr bool has_null = true;
};
*/

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
      auto ptr = (child_type *)Managers::get<hashed>()
                     ->get_allocator()
                     ->allocate_node(index)
                     ->ptr;
      data.insert(std::make_pair(i, ptr));
    }
  }

  TC_DEVICE TC_FORCE_INLINE int get_n() const {
    return data.size();
  }

  static constexpr bool has_null = true;
};

#if __CUDA_ARCH__
template <typename T>
__device__ bool unique_in_warp(T val) {
  auto mask = __activemask();

  auto warpId = threadIdx.x % warpSize;

  bool has_following_eqiv = 0;
  for (int i = 1; i < warpSize; i++) {
    auto cond = warpId + i < warpSize;
    bool same = (cond & (val == __shfl_down_sync(mask, val, i)));
    has_following_eqiv = has_following_eqiv || (cond && same);
  }

  return !has_following_eqiv;
}

__device__ int elect_leader(int mask) {
  return __ffs(mask) - 1;
}
#endif

template <typename _child_type>
struct pointer {
  using child_type = _child_type;
  child_type *data;
  int lock;
  // std::mutex mut;

  TC_DEVICE TC_FORCE_INLINE child_type *look_up(
      int i) {  // i is flattened index
    // TC_ASSERT(i == 0);
    TC_ASSERT(data != nullptr);
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
    if (data == nullptr) {
#if defined(__CUDA_ARCH__)
      int warpId = threadIdx.x % warpSize;
      int mask = __activemask();
      int uniques = __ballot_sync(mask, unique_in_warp((long long)&lock));
      while (uniques) {
        int leader = elect_leader(uniques);
        if (warpId == leader && data == nullptr) {
          while (atomicCAS(&lock, 0, 1) == 1)
            ;
          __threadfence();
#endif
          if (data == nullptr) {
            auto meta = Managers::get_instance()
                            ->get<pointer>()
                            ->get_allocator()
                            ->allocate_node(index);
            data = (child_type *)meta->ptr;
            meta->snode_ptr = (void **)(&data);
#if defined(__CUDA_ARCH__)
            __threadfence();
#endif
          }
#if defined(__CUDA_ARCH__)
          atomicExch(&lock, 0);
        }
        uniques ^= 1 << leader;
      }
#endif
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
#if defined(TL_HOST)
    // assuming serial
    n = std::max(n, i + 1);
#else
    TC_ASSERT(i < n);
#endif
    return &data[i];
  }

  __device__ TC_FORCE_INLINE void clear() {
    n = 0;
  }

  __device__ __host__ TC_FORCE_INLINE void append(child_type t) {
    auto tail = atomic_add(&n, 1);
    TC_ASSERT(tail < max_n);
    data[tail] = t;
  }

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
#if defined(TL_HOST)
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
