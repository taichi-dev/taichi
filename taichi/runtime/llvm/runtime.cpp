// This file will only be compiled into llvm bitcode by clang.
// The generated bitcode will likely get inlined for performance.

#if !defined(TI_INCLUDED) || !defined(_WIN32)

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <cstring>

#include "taichi/inc/constants.h"
#include "taichi/math/arithmetic.h"

struct Context;
using assert_failed_type = void (*)(const char *);
using host_printf_type = void (*)(const char *, ...);
using host_vsnprintf_type = int (*)(char *,
                                    std::size_t,
                                    const char *,
                                    std::va_list);
using vm_allocator_type = void *(*)(void *, std::size_t, std::size_t);
using RangeForTaskFunc = void(Context *, const char *tls, int i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int i));

#if defined(__linux__) && !ARCH_cuda && defined(TI_ARCH_x64)
__asm__(".symver logf,logf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
__asm__(".symver expf,expf@GLIBC_2.2.5");
#endif

#define STRUCT_FIELD(S, F)                              \
  extern "C" decltype(S::F) S##_get_##F(S *s) {         \
    return s->F;                                        \
  }                                                     \
  extern "C" decltype(S::F) *S##_get_ptr_##F(S *s) {    \
    return &(s->F);                                     \
  }                                                     \
  extern "C" void S##_set_##F(S *s, decltype(S::F) f) { \
    s->F = f;                                           \
  }

#define STRUCT_FIELD_ARRAY(S, F)                                             \
  extern "C" std::remove_all_extents_t<decltype(S::F)> S##_get_##F(S *s,     \
                                                                   int i) {  \
    return s->F[i];                                                          \
  }                                                                          \
  extern "C" void S##_set_##F(S *s, int i,                                   \
                              std::remove_all_extents_t<decltype(S::F)> f) { \
    s->F[i] = f;                                                             \
  };

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using float32 = float;
using float64 = double;

using i8 = int8;
using i32 = int32;
using i64 = int64;
using u8 = uint8;
using u32 = uint32;
using u64 = uint64;
using f32 = float32;
using f64 = float64;

using uint8 = uint8_t;
using Ptr = uint8 *;

using ContextArgType = long long;

#if ARCH_cuda
extern "C" {

void __assertfail(const char *message,
                  const char *file,
                  i32 line,
                  const char *function,
                  std::size_t charSize);
};
#endif

template <typename T>
void locked_task(void *lock, const T &func);

template <typename T, typename G>
void locked_task(void *lock, const T &func, const G &test);

template <typename T>
T ifloordiv(T a, T b) {
  auto r = a / b;
  // simply `a * b < 0` may leads to overflow (#969)
  //
  // Formal Anti-Regression Verification (FARV):
  //
  // old = a * b < 0
  // new = (a < 0) != (b < 0) && a
  //
  //  a  b old new
  //  -  -  f = f (f&t)
  //  -  +  t = t (t&t)
  //  0  -  f = f (t&f)
  //  0  +  f = f (f&f)
  //  +  -  t = t (t&t)
  //  +  +  f = f (f&t)
  //
  // the situation of `b = 0` is ignored since we get FPE anyway.
  //
  r -= T((a < 0) != (b < 0) && a && b * r != a);
  return r;
}

struct LLVMRuntime;
template <typename... Args>
void taichi_printf(LLVMRuntime *runtime, const char *format, Args &&... args);

extern "C" {

i64 cuda_clock_i64() {
  return 0;
}

void system_memfence() {
}

#if ARCH_cuda
void cuda_vprintf(Ptr format, Ptr arg);
#endif

#define DEFINE_UNARY_REAL_FUNC(F) \
  f32 F##_f32(f32 x) {            \
    return std::F(x);             \
  }                               \
  f64 F##_f64(f64 x) {            \
    return std::F(x);             \
  }

DEFINE_UNARY_REAL_FUNC(exp)
DEFINE_UNARY_REAL_FUNC(log)
DEFINE_UNARY_REAL_FUNC(tan)
DEFINE_UNARY_REAL_FUNC(tanh)
DEFINE_UNARY_REAL_FUNC(abs)
DEFINE_UNARY_REAL_FUNC(acos)
DEFINE_UNARY_REAL_FUNC(asin)
DEFINE_UNARY_REAL_FUNC(cos)
DEFINE_UNARY_REAL_FUNC(sin)

int abs_i32(int a) {
  if (a > 0) {
    return a;
  } else {
    return -a;
  }
}

#if ARCH_x64 || ARCH_arm64

u32 rand_u32() {
  static u32 x = 123456789, y = 362436069, z = 521288629, w = 88675123;
  u32 t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

u64 rand_u64() {
  return ((u64)rand_u32() << 32) + rand_u32();
}

f32 rand_f32() {
  return rand_u32() * f32(1 / 4294967296.0);
}

f64 rand_f64() {
  return rand_f32();
}

i32 rand_i32() {
  return rand_u32();
}

i64 rand_i64() {
  return rand_u64();
}

#endif

i32 floordiv_i32(i32 a, i32 b) {
  return ifloordiv(a, b);
}

i64 floordiv_i64(i64 a, i64 b) {
  return ifloordiv(a, b);
}

int min_i32(i32 a, i32 b) {
  return a < b ? a : b;
}

int min_i64(i64 a, i64 b) {
  return a < b ? a : b;
}

int max_i32(i32 a, i32 b) {
  return a > b ? a : b;
}

int max_i64(i64 a, i64 b) {
  return a > b ? a : b;
}

int32 logic_not_i32(int32 a) {
  return !a;
}

float32 sgn_f32(float32 a) {
  float32 b;
  if (a > 0)
    b = 1;
  else if (a < 0)
    b = -1;
  else
    b = 0;
  return b;
}

float64 sgn_f64(float64 a) {
  float32 b;
  if (a > 0)
    b = 1;
  else if (a < 0)
    b = -1;
  else
    b = 0;
  return b;
}

f32 atan2_f32(f32 a, f32 b) {
  return std::atan2(a, b);
}

f64 atan2_f64(f64 a, f64 b) {
  return std::atan2(a, b);
}

i32 pow_i32(i32 x, i32 n) {
  i32 tmp = x;
  i32 ans = 1;
  while (n) {
    if (n & 1)
      ans *= tmp;
    tmp *= tmp;
    n >>= 1;
  }
  return ans;
}

i64 pow_i64(i64 x, i64 n) {
  i64 tmp = x;
  i64 ans = 1;
  while (n) {
    if (n & 1)
      ans *= tmp;
    tmp *= tmp;
    n >>= 1;
  }
  return ans;
}

f32 pow_f32(f32 a, f32 b) {
  return std::pow(a, b);
}

f64 pow_f64(f64 a, f64 b) {
  return std::pow(a, b);
}

f32 __nv_sgnf(f32 x) {
  return sgn_f32(x);
}

f64 __nv_sgn(f64 x) {
  return sgn_f64(x);
}

struct PhysicalCoordinates {
  i32 val[taichi_max_num_indices];
};

STRUCT_FIELD_ARRAY(PhysicalCoordinates, val);

#include "context.h"

STRUCT_FIELD_ARRAY(Context, args);
STRUCT_FIELD(Context, runtime);

int32 Context_get_extra_args(Context *ctx, int32 i, int32 j) {
  return ctx->extra_args[i][j];
}

#include "atomic.h"

// These structures are accessible by both the LLVM backend and this C++ runtime
// file here (for building complex runtime functions in C++)

// These structs contain some "template parameters"

// Common Attributes
struct StructMeta {
  i32 snode_id;
  std::size_t element_size;
  i32 max_num_elements;

  Ptr (*lookup_element)(Ptr, Ptr, int i);

  Ptr (*from_parent_element)(Ptr);

  i32 (*is_active)(Ptr, Ptr, int i);

  i32 (*get_num_elements)(Ptr, Ptr);

  void (*refine_coordinates)(PhysicalCoordinates *inp_coord,
                             PhysicalCoordinates *refined_coord,
                             int index);

  Context *context;
};

STRUCT_FIELD(StructMeta, snode_id)
STRUCT_FIELD(StructMeta, element_size)
STRUCT_FIELD(StructMeta, max_num_elements)
STRUCT_FIELD(StructMeta, get_num_elements);
STRUCT_FIELD(StructMeta, lookup_element);
STRUCT_FIELD(StructMeta, from_parent_element);
STRUCT_FIELD(StructMeta, refine_coordinates);
STRUCT_FIELD(StructMeta, is_active);
STRUCT_FIELD(StructMeta, context);

struct LLVMRuntime;

constexpr bool enable_assert = true;

void taichi_assert(Context *context, i32 test, const char *msg);
void taichi_assert_runtime(LLVMRuntime *runtime, i32 test, const char *msg);
#define TI_ASSERT_INFO(x, msg) taichi_assert(context, (int)(x), msg)
#define TI_ASSERT(x) TI_ASSERT_INFO(x, #x)

void ___stubs___() {
#if ARCH_cuda
  cuda_vprintf(nullptr, nullptr);
  cuda_clock_i64();
#endif
}
}

bool is_power_of_two(uint32 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

/*
A simple list data structure that is infinitely long.
Data are organized in chunks, where each chunk is allocated on demand.
*/

struct ListManager {
  static constexpr std::size_t max_num_chunks = 1024;
  Ptr chunks[max_num_chunks];
  std::size_t element_size{0};
  std::size_t max_num_elements_per_chunk;
  i32 log2chunk_num_elements;
  i32 lock;
  i32 num_elements;
  LLVMRuntime *runtime;

  ListManager(LLVMRuntime *runtime,
              std::size_t element_size,
              std::size_t num_elements_per_chunk)
      : element_size(element_size),
        max_num_elements_per_chunk(num_elements_per_chunk),
        runtime(runtime) {
    taichi_assert_runtime(runtime, is_power_of_two(max_num_elements_per_chunk),
                          "max_num_elements_per_chunk must be POT.");
    lock = 0;
    num_elements = 0;
    log2chunk_num_elements = taichi::log2int(num_elements_per_chunk);
  }

  void append(void *data_ptr);

  i32 reserve_new_element() {
    auto i = atomic_add_i32(&num_elements, 1);
    auto chunk_id = i >> log2chunk_num_elements;
    touch_chunk(chunk_id);
    return i;
  }

  template <typename T>
  void push_back(const T &t) {
    this->append((void *)&t);
  }

  Ptr allocate();

  void touch_chunk(int chunk_id);

  i32 get_num_active_chunks() {
    i32 counter = 0;
    for (int i = 0; i < max_num_chunks; i++) {
      counter += (chunks[i] != nullptr);
    }
    return counter;
  }

  void clear() {
    num_elements = 0;
  }

  void resize(i32 n) {
    num_elements = n;
  }

  Ptr get_element_ptr(i32 i) {
    return chunks[i >> log2chunk_num_elements] +
           element_size * (i & ((1 << log2chunk_num_elements) - 1));
  }

  template <typename T>
  T &get(i32 i) {
    return *(T *)get_element_ptr(i);
  }

  Ptr touch_and_get(i32 i) {
    touch_chunk(i >> log2chunk_num_elements);
    return get_element_ptr(i);
  }

  i32 size() {
    return num_elements;
  }

  i32 ptr2index(Ptr ptr) {
    auto chunk_size = max_num_elements_per_chunk * element_size;
    for (int i = 0; i < max_num_chunks; i++) {
      taichi_assert_runtime(runtime, chunks[i] != nullptr, "ptr not found.");
      if (chunks[i] <= ptr && ptr < chunks[i] + chunk_size) {
        return (i << log2chunk_num_elements) +
               i32((ptr - chunks[i]) / element_size);
      }
    }
    return -1;
  }
};

STRUCT_FIELD(ListManager, element_size);
STRUCT_FIELD(ListManager, max_num_elements_per_chunk);
STRUCT_FIELD(ListManager, num_elements);

extern "C" {

struct Element {
  Ptr element;
  int loop_bounds[2];
  PhysicalCoordinates pcoord;
};

STRUCT_FIELD(Element, element);
STRUCT_FIELD(Element, pcoord);
STRUCT_FIELD_ARRAY(Element, loop_bounds);

struct RandState {
  u32 x;
  u32 y;
  u32 z;
  u32 w;
  i32 lock;
};

void initialize_rand_state(RandState *state, u32 i) {
  state->x = 123456789 * i * 1000000007;
  state->y = 362436069;
  state->z = 521288629;
  state->w = 88675123;
  state->lock = 0;
}
}

struct NodeManager;

struct LLVMRuntime {
  bool preallocated;
  std::size_t preallocated_size;

  Ptr preallocated_head;
  Ptr preallocated_tail;

  vm_allocator_type vm_allocator;
  assert_failed_type assert_failed;
  host_printf_type host_printf;
  host_vsnprintf_type host_vsnprintf;
  Ptr prog;
  Ptr root;
  size_t root_mem_size;
  Ptr thread_pool;
  parallel_for_type parallel_for;
  ListManager *element_lists[taichi_max_num_snodes];
  NodeManager *node_allocators[taichi_max_num_snodes];
  Ptr ambient_elements[taichi_max_num_snodes];
  Ptr temporaries;
  RandState *rand_states;
  MemRequestQueue *mem_req_queue;
  Ptr allocate(std::size_t size);
  Ptr allocate_aligned(std::size_t size, std::size_t alignment);
  Ptr request_allocate_aligned(std::size_t size, std::size_t alignment);
  Ptr allocate_from_buffer(std::size_t size, std::size_t alignment);
  Ptr profiler;
  void (*profiler_start)(Ptr, Ptr);
  void (*profiler_stop)(Ptr);

  char error_message_template[taichi_error_message_max_length];
  uint64 error_message_arguments[taichi_error_message_max_num_arguments];
  i32 error_message_lock = 0;
  i64 error_code = 0;

  Ptr result_buffer;
  i32 allocator_lock;

  i32 num_rand_states;

  i64 total_requested_memory;

  template <typename T>
  void set_result(std::size_t i, T t) {
    static_assert(sizeof(T) <= sizeof(uint64));
    ((u64 *)result_buffer)[i] =
        taichi_union_cast_with_different_sizes<uint64>(t);
  }

  template <typename T, typename... Args>
  T *create(Args &&... args) {
    auto ptr = (T *)request_allocate_aligned(sizeof(T), 4096);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }
};

// TODO: are these necessary?
STRUCT_FIELD_ARRAY(LLVMRuntime, element_lists);
STRUCT_FIELD_ARRAY(LLVMRuntime, node_allocators);
STRUCT_FIELD(LLVMRuntime, root);
STRUCT_FIELD(LLVMRuntime, root_mem_size);
STRUCT_FIELD(LLVMRuntime, temporaries);
STRUCT_FIELD(LLVMRuntime, assert_failed);
STRUCT_FIELD(LLVMRuntime, host_printf);
STRUCT_FIELD(LLVMRuntime, host_vsnprintf);
STRUCT_FIELD(LLVMRuntime, profiler);
STRUCT_FIELD(LLVMRuntime, profiler_start);
STRUCT_FIELD(LLVMRuntime, profiler_stop);

// NodeManager of node S (hash, pointer) managers the memory allocation of S_ch
// It makes use of three ListManagers.
struct NodeManager {
  LLVMRuntime *runtime;
  i32 lock;

  i32 element_size;
  i32 chunk_num_elements;
  i32 allocated_elements;

  ListManager *free_list, *recycled_list, *data_list;
  i32 recycle_list_size_backup;

  using list_data_type = i32;

  NodeManager(LLVMRuntime *runtime,
              i32 element_size,
              i32 chunk_num_elements = -1)
      : runtime(runtime), element_size(element_size) {
    // 16K elements per chunk, by default
    if (chunk_num_elements == -1) {
      chunk_num_elements = 16 * 1024;
    }
    // Maximum chunk size = 128 MB
    while (chunk_num_elements > 1 &&
           (uint64)chunk_num_elements * element_size > 128UL * 1024 * 1024) {
      chunk_num_elements /= 2;
    }
    this->chunk_num_elements = chunk_num_elements;
    allocated_elements = 0;
    free_list = runtime->create<ListManager>(runtime, sizeof(list_data_type),
                                             chunk_num_elements);
    recycled_list = runtime->create<ListManager>(
        runtime, sizeof(list_data_type), chunk_num_elements);
    data_list =
        runtime->create<ListManager>(runtime, element_size, chunk_num_elements);
  }

  Ptr allocate() {
    int old_cursor = atomic_add_i32(&allocated_elements, 1);
    i32 l;
    if (old_cursor >= free_list->size()) {
      // running out of free list. allocate new.
      l = data_list->reserve_new_element();
    } else {
      // reuse
      l = free_list->get<list_data_type>(old_cursor);
    }
    return data_list->get_element_ptr(l);
  }

  i32 locate(Ptr ptr) {
    return data_list->ptr2index(ptr);
  }

  void recycle(Ptr ptr) {
    auto index = locate(ptr);
    recycled_list->append(&index);
  }

  void gc_serial() {
    // compact free list
    for (int i = allocated_elements; i < free_list->size(); i++) {
      free_list->get<list_data_type>(i - allocated_elements) =
          free_list->get<list_data_type>(i);
    }
    const i32 num_unused = max_i32(free_list->size() - allocated_elements, 0);
    allocated_elements = 0;
    free_list->resize(num_unused);

    // zero-fill recycled and push to free list
    for (int i = 0; i < recycled_list->size(); i++) {
      auto idx = recycled_list->get<list_data_type>(i);
      auto ptr = data_list->get_element_ptr(idx);
      std::memset(ptr, 0, element_size);
      free_list->push_back(idx);
    }
    recycled_list->clear();
  }
};

extern "C" {

void LLVMRuntime_store_result(LLVMRuntime *runtime, u64 ret) {
  runtime->set_result(taichi_result_buffer_ret_value_id, ret);
}

void LLVMRuntime_profiler_start(LLVMRuntime *runtime, Ptr kernel_name) {
  runtime->profiler_start(runtime->profiler, kernel_name);
}

void LLVMRuntime_profiler_stop(LLVMRuntime *runtime) {
  runtime->profiler_stop(runtime->profiler);
}

Ptr get_temporary_pointer(LLVMRuntime *runtime, u64 offset) {
  return runtime->temporaries + offset;
}

void runtime_retrieve_and_reset_error_code(LLVMRuntime *runtime) {
  runtime->set_result(taichi_result_buffer_error_id, runtime->error_code);
  runtime->error_code = 0;
}

void runtime_retrieve_error_message(LLVMRuntime *runtime, int i) {
  runtime->set_result(taichi_result_buffer_error_id,
                      runtime->error_message_template[i]);
}

void runtime_retrieve_error_message_argument(LLVMRuntime *runtime,
                                             int argument_id) {
  runtime->set_result(taichi_result_buffer_error_id,
                      runtime->error_message_arguments[argument_id]);
}

void runtime_retrieve_element_list(LLVMRuntime *runtime, int snode_id) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      runtime->element_lists[snode_id]);
}

void runtime_element_list_retrieve_num_elements(LLVMRuntime *runtime,
                                                ListManager *list) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      list->num_elements);
}

void runtime_element_list_retrieve_element_size(LLVMRuntime *runtime,
                                                ListManager *list) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      list->element_size);
}

void runtime_element_list_retrieve_max_num_elements_per_chunk(
    LLVMRuntime *runtime,
    ListManager *list) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      list->max_num_elements_per_chunk);
}

void runtime_listmanager_get_num_active_chunks(LLVMRuntime *runtime,
                                               ListManager *list_manager) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      list_manager->get_num_active_chunks());
}

#define RUNTIME_STRUCT_FIELD(S, F)                                    \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s) { \
    runtime->set_result(taichi_result_buffer_runtime_query_id, s->F); \
  }

#define RUNTIME_STRUCT_FIELD_ARRAY(S, F)                                     \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s, int i) { \
    runtime->set_result(taichi_result_buffer_runtime_query_id, s->F[i]);     \
  }

RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, node_allocators);
RUNTIME_STRUCT_FIELD(LLVMRuntime, total_requested_memory);
RUNTIME_STRUCT_FIELD(NodeManager, free_list);
RUNTIME_STRUCT_FIELD(NodeManager, recycled_list);
RUNTIME_STRUCT_FIELD(NodeManager, data_list);
RUNTIME_STRUCT_FIELD(NodeManager, allocated_elements);

void taichi_assert(Context *context, i32 test, const char *msg) {
  taichi_assert_runtime(context->runtime, test, msg);
}

void taichi_assert_format(LLVMRuntime *runtime,
                          i32 test,
                          const char *format,
                          int num_arguments,
                          uint64 *arguments) {
  if (!enable_assert || test != 0)
    return;
  if (!runtime->error_code) {
    locked_task(&runtime->error_message_lock, [&] {
      if (!runtime->error_code) {
        runtime->error_code = 1;  // Assertion failure

        memset(runtime->error_message_template, 0,
               taichi_error_message_max_length);
        memcpy(runtime->error_message_template, format,
               std::min(strlen(format), taichi_error_message_max_length - 1));
        for (int i = 0; i < num_arguments; i++) {
          runtime->error_message_arguments[i] = arguments[i];
        }
      }
    });
  }
#if ARCH_cuda
  // Kill this CUDA thread.
  asm("exit;");
#else
  // TODO: kill this CPU thread here.
#endif
}

void taichi_assert_runtime(LLVMRuntime *runtime, i32 test, const char *msg) {
  taichi_assert_format(runtime, test, msg, 0, nullptr);
}

Ptr LLVMRuntime::allocate_aligned(std::size_t size, std::size_t alignment) {
  if (preallocated)
    return allocate_from_buffer(size, alignment);
  else
    return (Ptr)vm_allocator(prog, size, alignment);
}

Ptr LLVMRuntime::allocate_from_buffer(std::size_t size, std::size_t alignment) {
  Ptr ret;
  bool success = false;
  locked_task(&allocator_lock, [&] {
    auto alignment_bytes =
        alignment - 1 -
        ((std::size_t)preallocated_head + alignment - 1) % alignment;
    size += alignment_bytes;
    if (preallocated_head + size <= preallocated_tail) {
      ret = preallocated_head;
      preallocated_head += size;
      success = true;
    } else {
      success = false;
    }
  });
  if (!success) {
#if ARCH_cuda
    // Here unfortunately we have to rely on a native CUDA assert failure to
    // halt the whole grid. Using a taichi_assert_runtime will not finish the
    // whole kernel execution immediately.
    __assertfail("Out of CUDA pre-allocated memory", "Taichi JIT", 0,
                 "allocate_from_buffer", 1);
#endif
  }
  taichi_assert_runtime(this, success, "Out of pre-allocated memory");
  return ret;
}

Ptr LLVMRuntime::allocate(std::size_t size) {
  return allocate_aligned(size, 1);
}

Ptr LLVMRuntime::request_allocate_aligned(std::size_t size,
                                          std::size_t alignment) {
  atomic_add_i64(&total_requested_memory, size);
  if (preallocated)
    return allocate_from_buffer(size, alignment);
  else {
    auto i = atomic_add_i32(&mem_req_queue->tail, 1);
    taichi_assert_runtime(this, i <= taichi_max_num_mem_requests,
                          "Too many memory allocation requests.");
    auto volatile r = &mem_req_queue->requests[i];
    atomic_exchange_u64((uint64 *)&r->size, size);
    atomic_exchange_u64((uint64 *)&r->alignment, alignment);

    // wait for host to allocate
    while (r->ptr == nullptr) {
#if defined(ARCH_cuda)
      system_memfence();
#endif
    };
    return r->ptr;
  }
}

void runtime_get_mem_req_queue(LLVMRuntime *runtime) {
  runtime->set_result(taichi_result_buffer_ret_value_id,
                      runtime->mem_req_queue);
}

void runtime_initialize(
    Ptr result_buffer,
    Ptr prog,
    std::size_t root_size,
    std::size_t
        preallocated_size,  // Non-zero means use the preallocated buffer
    Ptr preallocated_buffer,
    i32 num_rand_states,
    void *_vm_allocator,
    void *_host_printf,
    void *_host_vsnprintf) {
  // bootstrap
  auto vm_allocator = (vm_allocator_type)_vm_allocator;
  auto host_printf = (host_printf_type)_host_printf;
  auto host_vsnprintf = (host_vsnprintf_type)_host_vsnprintf;
  LLVMRuntime *runtime = nullptr;
  Ptr preallocated_tail = preallocated_buffer + preallocated_size;
  if (preallocated_size) {
    runtime = (LLVMRuntime *)preallocated_buffer;
    preallocated_buffer +=
        taichi::iroundup(sizeof(LLVMRuntime), taichi_page_size);
  } else {
    runtime = (LLVMRuntime *)vm_allocator(prog, sizeof(LLVMRuntime), 128);
  }

  runtime->root_mem_size =
      taichi::iroundup((size_t)root_size, taichi_page_size);

  runtime->preallocated = preallocated_size > 0;
  runtime->preallocated_head = preallocated_buffer;
  runtime->preallocated_tail = preallocated_tail;

  runtime->result_buffer = result_buffer;
  runtime->set_result(taichi_result_buffer_ret_value_id, runtime);
  runtime->vm_allocator = vm_allocator;
  runtime->host_printf = host_printf;
  runtime->host_vsnprintf = host_vsnprintf;
  runtime->prog = prog;

  runtime->total_requested_memory = 0;

  // runtime->allocate ready to use
  runtime->mem_req_queue = (MemRequestQueue *)runtime->allocate_aligned(
      sizeof(MemRequestQueue), taichi_page_size);

  // For Metal runtime, we have to make sure that both the beginning address
  // and the size of the root buffer memory are aligned to page size.
  runtime->root_mem_size =
      taichi::iroundup((size_t)root_size, taichi_page_size);
  runtime->root =
      runtime->allocate_aligned(runtime->root_mem_size, taichi_page_size);

  runtime->temporaries = (Ptr)runtime->allocate_aligned(
      taichi_global_tmp_buffer_size, taichi_page_size);

  runtime->num_rand_states = num_rand_states;
  runtime->rand_states = (RandState *)runtime->allocate_aligned(
      sizeof(RandState) * runtime->num_rand_states, taichi_page_size);
  for (int i = 0; i < runtime->num_rand_states; i++)
    initialize_rand_state(&runtime->rand_states[i], i);
}

void runtime_initialize2(LLVMRuntime *runtime, int root_id, int num_snodes) {
  // runtime->request_allocate_aligned ready to use

  // initialize the root node element list
  for (int i = 0; i < num_snodes; i++) {
    // TODO: some SNodes do not actually need an element list.
    runtime->element_lists[i] =
        runtime->create<ListManager>(runtime, sizeof(Element), 1024 * 64);
  }
  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = runtime->root;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    elem.pcoord.val[i] = 0;
  }

  runtime->element_lists[root_id]->append(&elem);
}

void LLVMRuntime_initialize_thread_pool(LLVMRuntime *runtime,
                                        void *thread_pool,
                                        void *parallel_for) {
  runtime->thread_pool = (Ptr)thread_pool;
  runtime->parallel_for = (parallel_for_type)parallel_for;
}

void runtime_NodeAllocator_initialize(LLVMRuntime *runtime,
                                      int snode_id,
                                      std::size_t node_size) {
  runtime->node_allocators[snode_id] =
      runtime->create<NodeManager>(runtime, node_size, 1024 * 16);
}

void runtime_allocate_ambient(LLVMRuntime *runtime,
                              int snode_id,
                              std::size_t size) {
  // Do not use NodeManager for the ambient node since it will never be garbage
  // collected.
  runtime->ambient_elements[snode_id] =
      runtime->request_allocate_aligned(size, 128);
}

void mutex_lock_i32(Ptr mutex) {
  while (atomic_exchange_i32((i32 *)mutex, 1) == 1)
    ;
}

void mutex_unlock_i32(Ptr mutex) {
  atomic_exchange_i32((i32 *)mutex, 0);
}

int32 thread_idx() {
  return 0;
}

i32 warp_size() {
  return 32;
}

i32 warp_idx() {
  return thread_idx() % warp_size();
}

int32 block_idx() {
  return 0;
}

int32 block_dim() {
  return 0;
}

int32 ctlz_i32(i32 val) {
  return 0;
}

int32 cttz_i32(i32 val) {
  return 0;
}

int32 cuda_compute_capability() {
  return 0;
}

int32 cuda_ballot(bool bit) {
  return 0;
}

i32 cuda_shfl_down_sync_i32(u32 mask, i32 delta, i32 val, int width) {
  return 0;
}

i32 cuda_shfl_down_i32(i32 delta, i32 val, int width) {
  return 0;
}

int32 cuda_ballot_sync(int32 mask, bool bit) {
  return 0;
}

i32 cuda_match_any_sync_i32(i32 mask, i32 value) {
  return 0;
}

i32 cuda_match_any_sync_i64(i32 mask, i64 value) {
#if ARCH_cuda
  u32 ret;
  asm volatile("match.any.sync.b64  %0, %1, %2;"
               : "=r"(ret)
               : "l"(value), "r"(mask));
  return ret;
#else
  return 0;
#endif
}

#if ARCH_cuda
uint32 cuda_active_mask() {
  unsigned int mask;
  asm volatile("activemask.b32 %0;" : "=r"(mask));
  return mask;
}
#else
uint32 cuda_active_mask() {
  return 0;
}
#endif

int32 grid_dim() {
  return 0;
}

void block_barrier() {
}

void warp_barrier(uint32 mask) {
}

void block_memfence() {
}

void grid_memfence() {
}

// "Element", "component" are different concepts

void clear_list(LLVMRuntime *runtime, StructMeta *parent, StructMeta *child) {
  auto child_list = runtime->element_lists[child->snode_id];
  child_list->clear();
}

/*
 * The element list of a SNode, maintains pointers to its instances, and
 * instances' parents' coordinates
 */

// For the root node there is only one container,
// therefore we use a special kernel for more parallelism.
void element_listgen_root(LLVMRuntime *runtime,
                          StructMeta *parent,
                          StructMeta *child) {
  // If there's just one element in the parent list, we need to use the blocks
  // (instead of threads) to split the parent container
  auto parent_list = runtime->element_lists[parent->snode_id];
  auto child_list = runtime->element_lists[child->snode_id];
  // Cache the func pointers here for better compiler optimization
  auto parent_refine_coordinates = parent->refine_coordinates;
  auto parent_lookup_element = parent->lookup_element;
  auto child_get_num_elements = child->get_num_elements;
  auto child_from_parent_element = child->from_parent_element;
#if ARCH_cuda
  // All blocks share the only root container, which has only one child
  // container.
  // Each thread processes a subset of the child container for more parallelism.
  int c_start = block_dim() * block_idx() + thread_idx();
  int c_step = grid_dim() * block_dim();
#else
  int c_start = 0;
  int c_step = 1;
#endif
  // Note that the root node has only one container, and the `element`
  // representing that single container has only one 'child':
  // element.loop_bounds[0] = 0 and element.loop_bounds[1] = 1
  // Therefore, compared with element_listgen_nonroot,
  // we need neither `i` to loop over the `elements`, nor `j` to
  // loop over the children.

  auto element = parent_list->get<Element>(0);

  PhysicalCoordinates refined_coord;
  parent_refine_coordinates(&element.pcoord, &refined_coord, 0);

  auto ch_element = parent_lookup_element((Ptr)parent, element.element, 0);
  ch_element = child_from_parent_element((Ptr)ch_element);
  auto ch_num_elements = child_get_num_elements((Ptr)child, ch_element);
  auto ch_element_size =
      std::min(ch_num_elements, taichi_listgen_max_element_size);

  // Here is a grid-stride loop.
  for (int c = c_start; c * ch_element_size < ch_num_elements; c += c_step) {
    Element elem;
    elem.element = ch_element;
    elem.loop_bounds[0] = c * ch_element_size;
    elem.loop_bounds[1] = std::min((c + 1) * ch_element_size, ch_num_elements);
    elem.pcoord = refined_coord;
    child_list->append(&elem);
  }
}

void element_listgen_nonroot(LLVMRuntime *runtime,
                             StructMeta *parent,
                             StructMeta *child) {
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->size();
  auto child_list = runtime->element_lists[child->snode_id];
  // Cache the func pointers here for better compiler optimization
  auto parent_refine_coordinates = parent->refine_coordinates;
  auto parent_is_active = parent->is_active;
  auto parent_lookup_element = parent->lookup_element;
  auto child_get_num_elements = child->get_num_elements;
  auto child_from_parent_element = child->from_parent_element;
#if ARCH_cuda
  // Each block processes a slice of a parent container
  int i_start = block_idx();
  int i_step = grid_dim();
  // Each thread processes an element of the parent container
  int j_start = thread_idx();
  int j_step = block_dim();
#else
  int i_start = 0;
  int i_step = 1;
  int j_start = 0;
  int j_step = 1;
#endif
  for (int i = i_start; i < num_parent_elements; i += i_step) {
    auto element = parent_list->get<Element>(i);
    int j_lower = element.loop_bounds[0] + j_start;
    int j_higher = element.loop_bounds[1];
    for (int j = j_lower; j < j_higher; j += j_step) {
      PhysicalCoordinates refined_coord;
      parent_refine_coordinates(&element.pcoord, &refined_coord, j);
      if (parent_is_active((Ptr)parent, element.element, j)) {
        auto ch_element =
            parent_lookup_element((Ptr)parent, element.element, j);
        ch_element = child_from_parent_element((Ptr)ch_element);
        auto ch_num_elements = child_get_num_elements((Ptr)child, ch_element);
        auto ch_element_size =
            std::min(ch_num_elements, taichi_listgen_max_element_size);
        for (int ch_lower = 0; ch_lower < ch_num_elements;
             ch_lower += ch_element_size) {
          Element elem;
          elem.element = ch_element;
          elem.loop_bounds[0] = ch_lower;
          elem.loop_bounds[1] =
              std::min(ch_lower + ch_element_size, ch_num_elements);
          elem.pcoord = refined_coord;
          child_list->append(&elem);
        }
      }
    }
  }
}

using BlockTask = void(Context *, Element *, int, int);

struct cpu_block_task_helper_context {
  Context *context;
  BlockTask *task;
  ListManager *list;
  int element_size;
  int element_split;
};

// TODO: To enforce inlining, we need to create in LLVM a new function that
// calls block_helper and the BLS xlogues, and pass that function to the
// scheduler.

// TODO: TLS should be directly passed to the scheduler, so that it lives
// with the threads (instead of blocks).

void block_helper(void *ctx_, int i) {
  auto ctx = (cpu_block_task_helper_context *)(ctx_);
  int element_id = i / ctx->element_split;
  int part_size = ctx->element_size / ctx->element_split;
  int part_id = i % ctx->element_split;
  auto &e = ctx->list->get<Element>(element_id);
  int lower = e.loop_bounds[0] + part_id * part_size;
  int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
  upper = std::min(upper, e.loop_bounds[1]);
  if (lower < upper) {
    (*ctx->task)(ctx->context, &ctx->list->get<Element>(element_id), lower,
                 upper);
  }
}

void parallel_struct_for(Context *context,
                         int snode_id,
                         int element_size,
                         int element_split,
                         BlockTask *task,
                         int num_threads) {
  auto list = (context->runtime)->element_lists[snode_id];
  auto list_tail = list->size();
#if ARCH_cuda
  int i = block_idx();
  // TODO: refactor element_split more systematically.
  element_split = 1;
  const auto part_size = element_size / element_split;
  while (true) {
    int element_id = i / element_split;
    if (element_id >= list_tail)
      break;
    auto part_id = i % element_split;
    auto &e = list->get<Element>(element_id);
    int lower = e.loop_bounds[0] + part_id * part_size;
    int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
    upper = std::min(upper, e.loop_bounds[1]);
    if (lower < upper)
      task(context, &list->get<Element>(element_id), lower, upper);
    i += grid_dim();
  }
#else
  cpu_block_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.list = list;
  ctx.element_size = element_size;
  ctx.element_split = element_split;
  // printf("size %d spilt %d tail %d\n", ctx.element_size, ctx.element_split,
  // list_tail);
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool, list_tail * element_split,
                        num_threads, &ctx, block_helper);
#endif
}

using range_for_xlogue = void (*)(Context *, /*TLS*/ char *tls_base);

struct range_task_helper_context {
  Context *context;
  range_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  range_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int begin;
  int end;
  int block_size;
  int step;
};

void cpu_parallel_range_for_task(void *range_context, int task_id) {
  auto ctx = *(range_task_helper_context *)range_context;
  alignas(8) char tls_buffer[ctx.tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (ctx.prologue)
    ctx.prologue(ctx.context, tls_ptr);
  if (ctx.step == 1) {
    int block_start = ctx.begin + task_id * ctx.block_size;
    int block_end = std::min(block_start + ctx.block_size, ctx.end);
    for (int i = block_start; i < block_end; i++) {
      ctx.body(ctx.context, tls_ptr, i);
    }
  } else if (ctx.step == -1) {
    int block_start = ctx.end - task_id * ctx.block_size;
    int block_end = std::max(ctx.begin, block_start * ctx.block_size);
    for (int i = block_start - 1; i >= block_end; i--) {
      ctx.body(ctx.context, tls_ptr, i);
    }
  }
  if (ctx.epilogue)
    ctx.epilogue(ctx.context, tls_ptr);
}

void cpu_parallel_range_for(Context *context,
                            int num_threads,
                            int begin,
                            int end,
                            int step,
                            int block_dim,
                            range_for_xlogue prologue,
                            RangeForTaskFunc *body,
                            range_for_xlogue epilogue,
                            std::size_t tls_size) {
  range_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.begin = begin;
  ctx.end = end;
  ctx.step = step;
  if (step != 1 && step != -1) {
    taichi_printf(context->runtime, "step must not be %d\n", step);
    exit(-1);
  }
  if (block_dim == 0) {
    // adaptive block dim
    auto num_items = (ctx.end - ctx.begin) / std::abs(step);
    // ensure each thread has at least ~32 tasks for load balancing
    // and each task has at least 512 items to amortize scheduler overhead
    block_dim = std::min(512, std::max(1, num_items / (num_threads * 32)));
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool,
                        (end - begin + block_dim - 1) / block_dim, num_threads,
                        &ctx, cpu_parallel_range_for_task);
}

void gpu_parallel_range_for(Context *context,
                            int begin,
                            int end,
                            range_for_xlogue prologue,
                            RangeForTaskFunc *func,
                            range_for_xlogue epilogue,
                            const std::size_t tls_size) {
  int idx = thread_idx() + block_dim() * block_idx() + begin;
  alignas(8) char tls_buffer[tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (prologue)
    prologue(context, tls_ptr);
  while (idx < end) {
    func(context, tls_ptr, idx);
    idx += block_dim() * grid_dim();
  }
  if (epilogue)
    epilogue(context, tls_ptr);
}

i32 linear_thread_idx() {
  return block_idx() * block_dim() + thread_idx();
}

#include "node_dense.h"
#include "node_dynamic.h"
#include "node_pointer.h"
#include "node_root.h"
#include "node_bitmasked.h"

void ListManager::touch_chunk(int chunk_id) {
  if (!chunks[chunk_id]) {
    locked_task(&lock, [&] {
      // may have been allocated during lock contention
      if (!chunks[chunk_id]) {
        grid_memfence();
        auto chunk_ptr = runtime->request_allocate_aligned(
            max_num_elements_per_chunk * element_size, 4096);
        atomic_exchange_u64((u64 *)&chunks[chunk_id], (u64)chunk_ptr);
      }
    });
  }
}

void ListManager::append(void *data_ptr) {
  auto ptr = allocate();
  std::memcpy(ptr, data_ptr, element_size);
}

Ptr ListManager::allocate() {
  auto i = reserve_new_element();
  return get_element_ptr(i);
}

void node_gc(LLVMRuntime *runtime, int snode_id) {
  runtime->node_allocators[snode_id]->gc_serial();
}

void gc_parallel_0(LLVMRuntime *runtime, int snode_id) {
  auto allocator = runtime->node_allocators[snode_id];
  auto free_list = allocator->free_list;
  auto free_list_size = free_list->size();
  auto allocated_elements = allocator->allocated_elements;
  using T = NodeManager::list_data_type;

  // Move unused elements to the beginning of the free_list
  int i = linear_thread_idx();
  if (allocated_elements * 2 > free_list_size) {
    // Directly copy. Dst and src does not overlap
    auto items_to_copy = free_list_size - allocated_elements;
    while (i < items_to_copy) {
      free_list->get<T>(i) = free_list->get<T>(allocated_elements + i);
      i += grid_dim() * block_dim();
    }
  } else {
    // Move only non-overlapping parts
    auto items_to_copy = allocated_elements;
    while (i < items_to_copy) {
      free_list->get<T>(i) =
          free_list->get<T>(free_list_size - items_to_copy + i);
      i += grid_dim() * block_dim();
    }
  }
}

void gc_parallel_1(LLVMRuntime *runtime, int snode_id) {
  auto allocator = runtime->node_allocators[snode_id];
  auto free_list = allocator->free_list;

  const i32 num_unused = max_i32(free_list->size() - allocator->allocated_elements, 0);
  free_list->resize(num_unused);

  allocator->allocated_elements = 0;
  allocator->recycle_list_size_backup = allocator->recycled_list->size();
  allocator->recycled_list->clear();
}

void gc_parallel_2(LLVMRuntime *runtime, int snode_id) {
  auto allocator = runtime->node_allocators[snode_id];
  auto elements = allocator->recycle_list_size_backup;
  auto free_list = allocator->free_list;
  auto recycled_list = allocator->recycled_list;
  auto data_list = allocator->data_list;
  auto element_size = allocator->element_size;
  using T = NodeManager::list_data_type;
  auto i = block_idx();
  while (i < elements) {
    auto idx = recycled_list->get<T>(i);
    auto ptr = data_list->get_element_ptr(idx);
    if (thread_idx() == 0) {
      free_list->push_back(idx);
    }
    // memset
    auto ptr_stop = ptr + element_size;
    if ((uint64)ptr % 4 != 0) {
      auto new_ptr = ptr + 4 - (uint64)ptr % 4;
      if (thread_idx() == 0) {
        for (uint8 *p = ptr; p < new_ptr; p++) {
          *p = 0;
        }
      }
      ptr = new_ptr;
    }
    // now ptr is a multiple of 4
    ptr += thread_idx() * sizeof(uint32);
    while (ptr + sizeof(uint32) <= ptr_stop) {
      *(uint32 *)ptr = 0;
      ptr += sizeof(uint32) * block_dim();
    }
    while (ptr < ptr_stop) {
      *ptr = 0;
      ptr++;
    }
    i += grid_dim();
  }
}
}

#if ARCH_cuda

extern "C" {

u32 cuda_rand_u32(Context *context) {
  auto state =
      &((LLVMRuntime *)context->runtime)->rand_states[linear_thread_idx()];

  auto &x = state->x;
  auto &y = state->y;
  auto &z = state->z;
  auto &w = state->w;
  auto t = x ^ (x << 11);

  x = y;
  y = z;
  z = w;
  w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));

  return w * 1000000007;  // multiply a prime number here is very necessary -
                          // it decorrelates streams of PRNGs
}

uint64 cuda_rand_u64(Context *context) {
  return ((u64)cuda_rand_u32(context) << 32) + cuda_rand_u32(context);
}

f32 cuda_rand_f32(Context *context) {
  return cuda_rand_u32(context) * (1.0f / 4294967296.0f);
}

f32 cuda_rand_f64(Context *context) {
  return cuda_rand_f32(context);
}

i32 cuda_rand_i32(Context *context) {
  return cuda_rand_u32(context);
}

i64 cuda_rand_i64(Context *context) {
  return cuda_rand_u64(context);
}
};
#endif

struct printf_helper {
  char buffer[1024];
  int tail;

  printf_helper() {
    std::memset(buffer, 0, sizeof(buffer));
    tail = 0;
  }

  void push_back() {
  }

  template <typename... Args, typename T>
  void push_back(T t, Args &&... args) {
    *(T *)&buffer[tail] = t;
    if (tail % sizeof(T) != 0)
      tail += sizeof(T) - tail % sizeof(T);
    // align
    tail += sizeof(T);
    if constexpr ((sizeof...(args)) != 0) {
      push_back(std::forward<Args>(args)...);
    }
  }

  Ptr ptr() {
    return (Ptr) & (buffer[0]);
  }
};

template <typename... Args>
void taichi_printf(LLVMRuntime *runtime, const char *format, Args &&... args) {
#if ARCH_cuda
  printf_helper helper;
  helper.push_back(std::forward<Args>(args)...);
  cuda_vprintf((Ptr)format, helper.ptr());
#else
  runtime->host_printf(format, args...);
#endif
}

#include "locked_task.h"

extern "C" {  // local stack operations

Ptr stack_top_primal(Ptr stack, std::size_t element_size) {
  auto n = *(u64 *)stack;
  return stack + sizeof(u64) + (n - 1) * 2 * element_size;
}

Ptr stack_top_adjoint(Ptr stack, std::size_t element_size) {
  return stack_top_primal(stack, element_size) + element_size;
}

void stack_init(Ptr stack) {
  *(u64 *)stack = 0;
}

void stack_pop(Ptr stack) {
  auto &n = *(u64 *)stack;
  n--;
}

void stack_push(Ptr stack, size_t max_num_elements, std::size_t element_size) {
  u64 &n = *(u64 *)stack;
  n += 1;
  // TODO: assert n <= max_elements
  std::memset(stack_top_primal(stack, element_size), 0, element_size * 2);
}

#include "internal_functions.h"
}

#endif
