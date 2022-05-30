// This file will only be compiled into llvm bitcode by clang.
// The generated bitcode will likely get inlined for performance.

#if !defined(TI_INCLUDED) || !defined(_WIN32)
// The latest MSVC(Visual Studio 2019 version 16.10.1, MSVC 14.29.30037)
// uses llvm-11 as requirements. Check this link for details:
// https://github.com/microsoft/STL/blob/1866b848f0175c3361a916680a4318e7f0cc5482/stl/inc/yvals_core.h#L550-L561.
// However, we use llvm-10 for now and building will fail due to clang version
// mismatch. Therefore, we workaround this problem by define such flag to skip
// the version check.
// NOTE(#2428)
#if defined(_WIN32) || defined(_WIN64)
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <cstring>

#include "taichi/inc/constants.h"
#include "taichi/inc/cuda_kernel_utils.inc.h"
#include "taichi/math/arithmetic.h"

struct RuntimeContext;
using assert_failed_type = void (*)(const char *);
using host_printf_type = void (*)(const char *, ...);
using host_vsnprintf_type = int (*)(char *,
                                    std::size_t,
                                    const char *,
                                    std::va_list);
using vm_allocator_type = void *(*)(void *, std::size_t, std::size_t);
using RangeForTaskFunc = void(RuntimeContext *, const char *tls, int i);
using MeshForTaskFunc = void(RuntimeContext *, const char *tls, uint32_t i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int thread_id, int i));

#if defined(__linux__) && !ARCH_cuda && defined(TI_ARCH_x64)
__asm__(".symver logf,logf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
__asm__(".symver expf,expf@GLIBC_2.2.5");
#endif

// For accessing struct fields
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

// For fetching struct fields from device to host
#define RUNTIME_STRUCT_FIELD(S, F)                                    \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s) { \
    runtime->set_result(taichi_result_buffer_runtime_query_id, s->F); \
  }

#define RUNTIME_STRUCT_FIELD_ARRAY(S, F)                                     \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s, int i) { \
    runtime->set_result(taichi_result_buffer_runtime_query_id, s->F[i]);     \
  }

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
using i16 = int16;
using i32 = int32;
using i64 = int64;
using u8 = uint8;
using u16 = uint16;
using u32 = uint32;
using u64 = uint64;
using f32 = float32;
using f64 = float64;

using uint8 = uint8_t;
using Ptr = uint8 *;

using RuntimeContextArgType = long long;

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
void taichi_printf(LLVMRuntime *runtime, const char *format, Args &&...args);

extern "C" {

// This is not really a runtime function. Include this in a function body to
// mark it as force no inline. Helpful when preventing inlining huge function
// bodies.
void mark_force_no_inline() {
}

i64 cuda_clock_i64() {
  return 0;
}

void system_memfence() {
}

#if ARCH_cuda
void cuda_vprintf(Ptr format, Ptr arg);
#endif

// Note that strlen is undefined on the CUDA backend, so we manually
// implement it here.
std::size_t taichi_strlen(const char *str) {
  std::size_t len = 0;
  for (auto p = str; *p; p++)
    len++;
  return len;
}

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

i32 floordiv_i32(i32 a, i32 b) {
  return ifloordiv(a, b);
}

i64 floordiv_i64(i64 a, i64 b) {
  return ifloordiv(a, b);
}

u16 min_u16(u16 a, u16 b) {
  return a < b ? a : b;
}

i16 min_i16(i16 a, i16 b) {
  return a < b ? a : b;
}

u32 min_u32(u32 a, u32 b) {
  return a < b ? a : b;
}

int min_i32(i32 a, i32 b) {
  return a < b ? a : b;
}

u64 min_u64(u64 a, u64 b) {
  return a < b ? a : b;
}

i64 min_i64(i64 a, i64 b) {
  return a < b ? a : b;
}

u16 max_u16(u16 a, u16 b) {
  return a > b ? a : b;
}

i16 max_i16(i16 a, i16 b) {
  return a > b ? a : b;
}

u32 max_u32(u32 a, u32 b) {
  return a > b ? a : b;
}

int max_i32(i32 a, i32 b) {
  return a > b ? a : b;
}

u64 max_u64(u64 a, u64 b) {
  return a > b ? a : b;
}

i64 max_i64(i64 a, i64 b) {
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

#include "taichi/program/context.h"
#include "taichi/runtime/llvm/mem_request.h"

STRUCT_FIELD_ARRAY(RuntimeContext, args);
STRUCT_FIELD(RuntimeContext, runtime);
STRUCT_FIELD(RuntimeContext, result_buffer)

int32 RuntimeContext_get_extra_args(RuntimeContext *ctx, int32 i, int32 j) {
  return ctx->extra_args[i][j];
}

#include "taichi/runtime/llvm/atomic.h"

// These structures are accessible by both the LLVM backend and this C++ runtime
// file here (for building complex runtime functions in C++)

// These structs contain some "template parameters"

// Common Attributes
struct StructMeta {
  i32 snode_id;
  std::size_t element_size;
  i64 max_num_elements;

  Ptr (*lookup_element)(Ptr, Ptr, int i);

  Ptr (*from_parent_element)(Ptr);

  i32 (*is_active)(Ptr, Ptr, int i);

  i32 (*get_num_elements)(Ptr, Ptr);

  void (*refine_coordinates)(PhysicalCoordinates *inp_coord,
                             PhysicalCoordinates *refined_coord,
                             int index);

  RuntimeContext *context;
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

void taichi_assert(RuntimeContext *context, i32 test, const char *msg);
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

// TODO: there are many i32 types in this class, which may be an issue if there
// are >= 2 ** 31 elements.
struct ListManager {
  static constexpr std::size_t max_num_chunks = 128 * 1024;
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
  Ptr memory_pool;

  Ptr roots[kMaxNumSnodeTreesLlvm];
  size_t root_mem_sizes[kMaxNumSnodeTreesLlvm];

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

  Ptr wasm_print_buffer = nullptr;

  template <typename T>
  void set_result(std::size_t i, T t) {
    static_assert(sizeof(T) <= sizeof(uint64));
    ((u64 *)result_buffer)[i] =
        taichi_union_cast_with_different_sizes<uint64>(t);
  }

  template <typename T, typename... Args>
  T *create(Args &&...args) {
    auto ptr = (T *)request_allocate_aligned(sizeof(T), 4096);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }
};

// TODO: are these necessary?
STRUCT_FIELD_ARRAY(LLVMRuntime, element_lists);
STRUCT_FIELD_ARRAY(LLVMRuntime, node_allocators);
STRUCT_FIELD_ARRAY(LLVMRuntime, roots);
STRUCT_FIELD_ARRAY(LLVMRuntime, root_mem_sizes);
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
  i32 free_list_used;

  ListManager *free_list, *recycled_list, *data_list;
  i32 recycle_list_size_backup;

  using list_data_type = i32;

  NodeManager(LLVMRuntime *runtime,
              i32 element_size,
              i32 chunk_num_elements = -1)
      : runtime(runtime), element_size(element_size) {
    // 128K elements per chunk, by default
    if (chunk_num_elements == -1) {
      chunk_num_elements = 128 * 1024;
    }
    // Maximum chunk size = 128 MB
    while (chunk_num_elements > 1 &&
           (uint64)chunk_num_elements * element_size > 128UL * 1024 * 1024) {
      chunk_num_elements /= 2;
    }
    this->chunk_num_elements = chunk_num_elements;
    free_list_used = 0;
    free_list = runtime->create<ListManager>(runtime, sizeof(list_data_type),
                                             chunk_num_elements);
    recycled_list = runtime->create<ListManager>(
        runtime, sizeof(list_data_type), chunk_num_elements);
    data_list =
        runtime->create<ListManager>(runtime, element_size, chunk_num_elements);
  }

  Ptr allocate() {
    int old_cursor = atomic_add_i32(&free_list_used, 1);
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
    for (int i = free_list_used; i < free_list->size(); i++) {
      free_list->get<list_data_type>(i - free_list_used) =
          free_list->get<list_data_type>(i);
    }
    const i32 num_unused = max_i32(free_list->size() - free_list_used, 0);
    free_list_used = 0;
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

void RuntimeContext_store_result(RuntimeContext *ctx, u64 ret, u32 idx) {
  ctx->result_buffer[taichi_result_buffer_ret_value_id + idx] = ret;
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

void runtime_ListManager_get_num_active_chunks(LLVMRuntime *runtime,
                                               ListManager *list_manager) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      list_manager->get_num_active_chunks());
}

RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, node_allocators);
RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, element_lists);
RUNTIME_STRUCT_FIELD(LLVMRuntime, total_requested_memory);

RUNTIME_STRUCT_FIELD(NodeManager, free_list);
RUNTIME_STRUCT_FIELD(NodeManager, recycled_list);
RUNTIME_STRUCT_FIELD(NodeManager, data_list);
RUNTIME_STRUCT_FIELD(NodeManager, free_list_used);

RUNTIME_STRUCT_FIELD(ListManager, num_elements);
RUNTIME_STRUCT_FIELD(ListManager, max_num_elements_per_chunk);
RUNTIME_STRUCT_FIELD(ListManager, element_size);

void taichi_assert(RuntimeContext *context, i32 test, const char *msg) {
  taichi_assert_runtime(context->runtime, test, msg);
}

void taichi_assert_format(LLVMRuntime *runtime,
                          i32 test,
                          const char *format,
                          int num_arguments,
                          uint64 *arguments) {
  mark_force_no_inline();

  if (!enable_assert || test != 0)
    return;
  if (!runtime->error_code) {
    locked_task(&runtime->error_message_lock, [&] {
      if (!runtime->error_code) {
        runtime->error_code = 1;  // Assertion failure

        memset(runtime->error_message_template, 0,
               taichi_error_message_max_length);
        memcpy(runtime->error_message_template, format,
               std::min(taichi_strlen(format),
                        taichi_error_message_max_length - 1));
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
  // TODO: properly kill this CPU thread here, considering the containing
  // ThreadPool structure.

  // std::terminate();

  // Note that std::terminate() will throw an signal 6
  // (Aborted), which will be caught by Taichi's signal handler. The assert
  // failure message will NOT be properly printed since Taichi exits after
  // receiving that signal. It is better than nothing when debugging the
  // runtime, since otherwise the whole program may crash if the kernel
  // continues after assertion failure.
#endif
}

void taichi_assert_runtime(LLVMRuntime *runtime, i32 test, const char *msg) {
  taichi_assert_format(runtime, test, msg, 0, nullptr);
}

Ptr LLVMRuntime::allocate_aligned(std::size_t size, std::size_t alignment) {
  if (preallocated) {
    return allocate_from_buffer(size, alignment);
  }
  return (Ptr)vm_allocator(memory_pool, size, alignment);
}

Ptr LLVMRuntime::allocate_from_buffer(std::size_t size, std::size_t alignment) {
  Ptr ret = nullptr;
  bool success = false;
  locked_task(&allocator_lock, [&] {
    auto alignment_bytes =
        alignment - 1 -
        ((std::size_t)preallocated_head + alignment - 1) % alignment;
    size += alignment_bytes;
    if (preallocated_head + size <= preallocated_tail) {
      ret = preallocated_head + alignment_bytes;
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
    __assertfail(
        "Out of CUDA pre-allocated memory.\n"
        "Consider using ti.init(device_memory_fraction=0.9) or "
        "ti.init(device_memory_GB=4) to allocate more"
        " GPU memory",
        "Taichi JIT", 0, "allocate_from_buffer", 1);
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

void runtime_memory_allocate_aligned(LLVMRuntime *runtime,
                                     std::size_t size,
                                     std::size_t alignment) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      runtime->allocate_aligned(size, alignment));
}

void runtime_get_mem_req_queue(LLVMRuntime *runtime) {
  runtime->set_result(taichi_result_buffer_ret_value_id,
                      runtime->mem_req_queue);
}

void runtime_initialize(
    Ptr result_buffer,
    Ptr memory_pool,
    std::size_t
        preallocated_size,  // Non-zero means use the preallocated buffer
    Ptr preallocated_buffer,
    i32 starting_rand_state,
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
    runtime =
        (LLVMRuntime *)vm_allocator(memory_pool, sizeof(LLVMRuntime), 128);
  }

  runtime->preallocated = preallocated_size > 0;
  runtime->preallocated_head = preallocated_buffer;
  runtime->preallocated_tail = preallocated_tail;

  runtime->result_buffer = result_buffer;
  runtime->set_result(taichi_result_buffer_ret_value_id, runtime);
  runtime->vm_allocator = vm_allocator;
  runtime->host_printf = host_printf;
  runtime->host_vsnprintf = host_vsnprintf;
  runtime->memory_pool = memory_pool;

  runtime->total_requested_memory = 0;

  // runtime->allocate ready to use
  runtime->mem_req_queue = (MemRequestQueue *)runtime->allocate_aligned(
      sizeof(MemRequestQueue), taichi_page_size);

  runtime->temporaries = (Ptr)runtime->allocate_aligned(
      taichi_global_tmp_buffer_size, taichi_page_size);

  runtime->num_rand_states = num_rand_states;
  runtime->rand_states = (RandState *)runtime->allocate_aligned(
      sizeof(RandState) * runtime->num_rand_states, taichi_page_size);
  for (int i = 0; i < runtime->num_rand_states; i++)
    initialize_rand_state(&runtime->rand_states[i], starting_rand_state + i);
}

void runtime_initialize_snodes(LLVMRuntime *runtime,
                               std::size_t root_size,
                               const int root_id,
                               const int num_snodes,
                               const int snode_tree_id,
                               std::size_t rounded_size,
                               Ptr ptr,
                               bool all_dense) {
  // For Metal runtime, we have to make sure that both the beginning address
  // and the size of the root buffer memory are aligned to page size.
  runtime->root_mem_sizes[snode_tree_id] = rounded_size;
  runtime->roots[snode_tree_id] = ptr;
  // runtime->request_allocate_aligned ready to use
  // initialize the root node element list
  if (all_dense) {
    return;
  }
  for (int i = root_id; i < root_id + num_snodes; i++) {
    // TODO: some SNodes do not actually need an element list.
    runtime->element_lists[i] =
        runtime->create<ListManager>(runtime, sizeof(Element), 1024 * 64);
  }
  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = runtime->roots[snode_tree_id];
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

i32 cuda_shfl_down_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_down_i32(i32 delta, i32 val, int width) {
  return 0;
}

f32 cuda_shfl_down_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_down_f32(i32 delta, f32 val, int width) {
  return 0;
}

i32 cuda_shfl_xor_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_up_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_up_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

bool cuda_all_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_all_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_all_sync(mask, (bool)predicate);
}

bool cuda_any_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_any_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_any_sync(mask, (bool)predicate);
}

bool cuda_uni_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_uni_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_uni_sync(mask, (bool)predicate);
}

int32 cuda_ballot_sync(int32 mask, bool bit) {
  return 0;
}

int32 cuda_ballot_i32(int32 predicate) {
  return cuda_ballot_sync(UINT32_MAX, (bool)predicate);
}

int32 cuda_ballot_sync_i32(u32 mask, int32 predicate) {
  return cuda_ballot_sync(mask, (bool)predicate);
}

uint32 cuda_match_any_sync_i32(u32 mask, i32 value) {
  return 0;
}

u32 cuda_match_all_sync_i32(u32 mask, i32 value) {
#if ARCH_cuda
  u32 ret;
  asm volatile("match.all.sync.b32  %0, %1, %2;"
               : "=r"(ret)
               : "r"(value), "r"(mask));
  return ret;
#else
  return 0;
#endif
}

uint32 cuda_match_any_sync_i64(u32 mask, i64 value) {
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

void block_barrier() {
}

void warp_barrier(uint32 mask) {
}

void block_memfence() {
}

void grid_memfence() {
}

// these trivial functions are needed by the DEFINE_REDUCTION macro
i32 op_add_i32(i32 a, i32 b) {
  return a + b;
}
f32 op_add_f32(f32 a, f32 b) {
  return a + b;
}

i32 op_min_i32(i32 a, i32 b) {
  return std::min(a, b);
}
f32 op_min_f32(f32 a, f32 b) {
  return std::min(a, b);
}

i32 op_max_i32(i32 a, i32 b) {
  return std::max(a, b);
}
f32 op_max_f32(f32 a, f32 b) {
  return std::max(a, b);
}

i32 op_and_i32(i32 a, i32 b) {
  return a & b;
}
i32 op_or_i32(i32 a, i32 b) {
  return a | b;
}
i32 op_xor_i32(i32 a, i32 b) {
  return a ^ b;
}

#define DEFINE_REDUCTION(op, dtype)                                       \
  dtype warp_reduce_##op##_##dtype(dtype val) {                           \
    for (int offset = 16; offset > 0; offset /= 2)                        \
      val = op_##op##_##dtype(                                            \
          val, cuda_shfl_down_sync_##dtype(0xFFFFFFFF, val, offset, 31)); \
    return val;                                                           \
  }                                                                       \
  dtype reduce_##op##_##dtype(dtype *result, dtype val) {                 \
    dtype warp_result = warp_reduce_##op##_##dtype(val);                  \
    if ((thread_idx() & (warp_size() - 1)) == 0) {                        \
      atomic_##op##_##dtype(result, warp_result);                         \
    }                                                                     \
    return val;                                                           \
  }

DEFINE_REDUCTION(add, i32);
DEFINE_REDUCTION(add, f32);

DEFINE_REDUCTION(min, i32);
DEFINE_REDUCTION(min, f32);

DEFINE_REDUCTION(max, i32);
DEFINE_REDUCTION(max, f32);

DEFINE_REDUCTION(and, i32);
DEFINE_REDUCTION(or, i32);
DEFINE_REDUCTION(xor, i32);

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
    // There is no need to refine coordinates for root listgen, since its
    // num_bits is always zero
    elem.pcoord = element.pcoord;
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

using BlockTask = void(RuntimeContext *, char *, Element *, int, int);

struct cpu_block_task_helper_context {
  RuntimeContext *context;
  BlockTask *task;
  ListManager *list;
  int element_size;
  int element_split;
  std::size_t tls_buffer_size;
};

// TODO: To enforce inlining, we need to create in LLVM a new function that
// calls block_helper and the BLS xlogues, and pass that function to the
// scheduler.

// TODO: TLS should be directly passed to the scheduler, so that it lives
// with the threads (instead of blocks).

void cpu_struct_for_block_helper(void *ctx_, int thread_id, int i) {
  auto ctx = (cpu_block_task_helper_context *)(ctx_);
  int element_id = i / ctx->element_split;
  int part_size = ctx->element_size / ctx->element_split;
  int part_id = i % ctx->element_split;
  auto &e = ctx->list->get<Element>(element_id);
  int lower = e.loop_bounds[0] + part_id * part_size;
  int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
  upper = std::min(upper, e.loop_bounds[1]);
  alignas(8) char tls_buffer[ctx->tls_buffer_size];

  RuntimeContext this_thread_context = *ctx->context;
  this_thread_context.cpu_thread_id = thread_id;
  if (lower < upper) {
    (*ctx->task)(&this_thread_context, tls_buffer,
                 &ctx->list->get<Element>(element_id), lower, upper);
  }
}

void parallel_struct_for(RuntimeContext *context,
                         int snode_id,
                         int element_size,
                         int element_split,
                         BlockTask *task,
                         std::size_t tls_buffer_size,
                         int num_threads) {
  auto list = (context->runtime)->element_lists[snode_id];
  auto list_tail = list->size();
#if ARCH_cuda
  int i = block_idx();
  // Note: CUDA requires compile-time constant local array sizes.
  // We use "1" here and modify it during codegen to tls_buffer_size.
  alignas(8) char tls_buffer[1];
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
      task(context, tls_buffer, &list->get<Element>(element_id), lower, upper);
    i += grid_dim();
  }
#else
  cpu_block_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.list = list;
  ctx.element_size = element_size;
  ctx.element_split = element_split;
  ctx.tls_buffer_size = tls_buffer_size;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool, list_tail * element_split,
                        num_threads, &ctx, cpu_struct_for_block_helper);
#endif
}

using range_for_xlogue = void (*)(RuntimeContext *, /*TLS*/ char *tls_base);
using mesh_for_xlogue = void (*)(RuntimeContext *,
                                 /*TLS*/ char *tls_base,
                                 uint32_t patch_idx);

struct range_task_helper_context {
  RuntimeContext *context;
  range_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  range_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int begin;
  int end;
  int block_size;
  int step;
};

void cpu_parallel_range_for_task(void *range_context,
                                 int thread_id,
                                 int task_id) {
  auto ctx = *(range_task_helper_context *)range_context;
  alignas(8) char tls_buffer[ctx.tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (ctx.prologue)
    ctx.prologue(ctx.context, tls_ptr);

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;
  if (ctx.step == 1) {
    int block_start = ctx.begin + task_id * ctx.block_size;
    int block_end = std::min(block_start + ctx.block_size, ctx.end);
    for (int i = block_start; i < block_end; i++) {
      ctx.body(&this_thread_context, tls_ptr, i);
    }
  } else if (ctx.step == -1) {
    int block_start = ctx.end - task_id * ctx.block_size;
    int block_end = std::max(ctx.begin, block_start * ctx.block_size);
    for (int i = block_start - 1; i >= block_end; i--) {
      ctx.body(&this_thread_context, tls_ptr, i);
    }
  }
  if (ctx.epilogue)
    ctx.epilogue(ctx.context, tls_ptr);
}

void cpu_parallel_range_for(RuntimeContext *context,
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

void gpu_parallel_range_for(RuntimeContext *context,
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

struct mesh_task_helper_context {
  RuntimeContext *context;
  mesh_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  mesh_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int num_patches;
  int block_size;
};

void cpu_parallel_mesh_for_task(void *range_context,
                                int thread_id,
                                int task_id) {
  auto ctx = *(mesh_task_helper_context *)range_context;
  alignas(8) char tls_buffer[ctx.tls_size];
  auto tls_ptr = &tls_buffer[0];

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;

  int block_start = task_id * ctx.block_size;
  int block_end = std::min(block_start + ctx.block_size, ctx.num_patches);

  for (int idx = block_start; idx < block_end; idx++) {
    if (ctx.prologue)
      ctx.prologue(ctx.context, tls_ptr, idx);
    ctx.body(&this_thread_context, tls_ptr, idx);
    if (ctx.epilogue)
      ctx.epilogue(ctx.context, tls_ptr, idx);
  }
}

void cpu_parallel_mesh_for(RuntimeContext *context,
                           int num_threads,
                           int num_patches,
                           int block_dim,
                           mesh_for_xlogue prologue,
                           RangeForTaskFunc *body,
                           mesh_for_xlogue epilogue,
                           std::size_t tls_size) {
  mesh_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.num_patches = num_patches;
  if (block_dim == 0) {
    // adaptive block dim
    // ensure each thread has at least ~32 tasks for load balancing
    // and each task has at least 512 items to amortize scheduler overhead
    block_dim = std::min(512, std::max(1, num_patches / (num_threads * 32)));
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool,
                        (num_patches + block_dim - 1) / block_dim, num_threads,
                        &ctx, cpu_parallel_mesh_for_task);
}

void gpu_parallel_mesh_for(RuntimeContext *context,
                           int num_patches,
                           mesh_for_xlogue prologue,
                           MeshForTaskFunc *func,
                           mesh_for_xlogue epilogue,
                           const std::size_t tls_size) {
  alignas(8) char tls_buffer[tls_size];
  auto tls_ptr = &tls_buffer[0];
  for (int idx = block_idx(); idx < num_patches; idx += grid_dim()) {
    if (prologue)
      prologue(context, tls_ptr, idx);
    func(context, tls_ptr, idx);
    if (epilogue)
      epilogue(context, tls_ptr, idx);
  }
}

i32 linear_thread_idx(RuntimeContext *context) {
#if ARCH_cuda
  return block_idx() * block_dim() + thread_idx();
#else
  return context->cpu_thread_id;
#endif
}

#include "node_dense.h"
#include "node_dynamic.h"
#include "node_pointer.h"
#include "node_root.h"
#include "node_bitmasked.h"

void ListManager::touch_chunk(int chunk_id) {
  taichi_assert_runtime(runtime, chunk_id < max_num_chunks,
                        "List manager out of chunks.");
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

void gc_parallel_0(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  auto allocator = runtime->node_allocators[snode_id];
  auto free_list = allocator->free_list;
  auto free_list_size = free_list->size();
  auto free_list_used = allocator->free_list_used;
  using T = NodeManager::list_data_type;

  // Move unused elements to the beginning of the free_list
  int i = linear_thread_idx(context);
  if (free_list_used * 2 > free_list_size) {
    // Directly copy. Dst and src does not overlap
    auto items_to_copy = free_list_size - free_list_used;
    while (i < items_to_copy) {
      free_list->get<T>(i) = free_list->get<T>(free_list_used + i);
      i += grid_dim() * block_dim();
    }
  } else {
    // Move only non-overlapping parts
    auto items_to_copy = free_list_used;
    while (i < items_to_copy) {
      free_list->get<T>(i) =
          free_list->get<T>(free_list_size - items_to_copy + i);
      i += grid_dim() * block_dim();
    }
  }
}

void gc_parallel_1(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  auto allocator = runtime->node_allocators[snode_id];
  auto free_list = allocator->free_list;

  const i32 num_unused =
      max_i32(free_list->size() - allocator->free_list_used, 0);
  free_list->resize(num_unused);

  allocator->free_list_used = 0;
  allocator->recycle_list_size_backup = allocator->recycled_list->size();
  allocator->recycled_list->clear();
}

void gc_parallel_2(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
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

extern "C" {

u32 rand_u32(RuntimeContext *context) {
  auto state = &((LLVMRuntime *)context->runtime)
                    ->rand_states[linear_thread_idx(context)];

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
                          // it decorrelates streams of PRNGs.
}

uint64 rand_u64(RuntimeContext *context) {
  return ((u64)rand_u32(context) << 32) + rand_u32(context);
}

f32 rand_f32(RuntimeContext *context) {
  return (rand_u32(context) >> 8) * (1.0f / 16777216.0f);
}

f64 rand_f64(RuntimeContext *context) {
  return (rand_u64(context) >> 11) * (1.0 / 9007199254740992.0);
}

i32 rand_i32(RuntimeContext *context) {
  return rand_u32(context);
}

i64 rand_i64(RuntimeContext *context) {
  return rand_u64(context);
}
};

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
  void push_back(T t, Args &&...args) {
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
void taichi_printf(LLVMRuntime *runtime, const char *format, Args &&...args) {
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

// TODO: make here less repetitious.
// Original implementation is
// u##N mask = ((((u##N)1 << bits) - 1) << offset);
// When N equals bits equals 32, 32 times of left shifting will be carried on
// which is an undefined behavior.
// see #2096 for more details
#define DEFINE_SET_PARTIAL_BITS(N)                                            \
  void set_mask_b##N(u##N *ptr, u64 mask, u##N value) {                       \
    u##N mask_N = (u##N)mask;                                                 \
    *ptr = (*ptr & (~mask_N)) | value;                                        \
  }                                                                           \
                                                                              \
  void atomic_set_mask_b##N(u##N *ptr, u64 mask, u##N value) {                \
    u##N mask_N = (u##N)mask;                                                 \
    u##N new_value = 0;                                                       \
    u##N old_value = *ptr;                                                    \
    do {                                                                      \
      old_value = *ptr;                                                       \
      new_value = (old_value & (~mask_N)) | value;                            \
    } while (                                                                 \
        !__atomic_compare_exchange(ptr, &old_value, &new_value, true,         \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
  }                                                                           \
                                                                              \
  void set_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits, u##N value) {   \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);            \
    set_mask_b##N(ptr, mask, value << offset);                                \
  }                                                                           \
                                                                              \
  void atomic_set_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits,          \
                                    u##N value) {                             \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);            \
    atomic_set_mask_b##N(ptr, mask, value << offset);                         \
  }                                                                           \
                                                                              \
  u##N atomic_add_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits,          \
                                    u##N value) {                             \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);            \
    u##N new_value = 0;                                                       \
    u##N old_value = *ptr;                                                    \
    do {                                                                      \
      old_value = *ptr;                                                       \
      new_value = old_value + (value << offset);                              \
      new_value = (old_value & (~mask)) | (new_value & mask);                 \
    } while (                                                                 \
        !__atomic_compare_exchange(ptr, &old_value, &new_value, true,         \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
    return old_value;                                                         \
  }

DEFINE_SET_PARTIAL_BITS(8);
DEFINE_SET_PARTIAL_BITS(16);
DEFINE_SET_PARTIAL_BITS(32);
DEFINE_SET_PARTIAL_BITS(64);

f32 rounding_prepare_f32(f32 f) {
  /* slower (but clearer) version with branching:
  if (f > 0)
    return f + 0.5;
  else
    return f - 0.5;
  */

  // Branch-free implementation: copy the sign bit of "f" to "0.5"
  i32 delta_bits =
      (taichi_union_cast<i32>(f) & 0x80000000) | taichi_union_cast<i32>(0.5f);
  f32 delta = taichi_union_cast<f32>(delta_bits);
  return f + delta;
}

f64 rounding_prepare_f64(f64 f) {
  // Same as above
  i64 delta_bits = (taichi_union_cast<i64>(f) & 0x8000000000000000LL) |
                   taichi_union_cast<i64>(0.5);
  f64 delta = taichi_union_cast<f64>(delta_bits);
  return f + delta;
}
}

namespace {
i32 kWasmPrintBufferSize = 1024 * 1024;
}

extern "C" {
// The input means starting address of RuntimeContext, which should be set to
// '__heap_base' to avoid conflicts with C++ stack data which is stored in
// memory. The function returns starting address of root buffer. The print
// buffer locates just before RuntimeContext (8MB). Here is an illustration for
// proper memory layout in WASM:
// clang-format off
// ━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━
//  Print ┃▄ RuntimeContext ┃  ▄ Runtime ▄  ┃ RandState[0] ┃ Root Buffer ...
// ━━━━━━━┻│━━━━━━━━━━━━━━━━▲━━│━━━━━━━━━│━━▲━━━━━━━━━━━━━━▲━━━━━━━━━━━━━━━━━━━
//         └────────────────┘  │         └──┘              │
//                             └───────────────────────────┘
// clang-format on
i32 wasm_materialize(RuntimeContext *context) {
  context->runtime = (LLVMRuntime *)((size_t)context + sizeof(RuntimeContext));
  context->runtime->rand_states =
      (RandState *)((size_t)context->runtime + sizeof(LLVMRuntime));
  // set random seed to (1, 0, 0, 0)
  context->runtime->rand_states[0].x = 1;
  // TODO: remove hard coding on root id 0(SNodeTree::kFirstID)
  context->runtime->roots[0] =
      (Ptr)((size_t)context->runtime->rand_states + sizeof(RandState));
  return (i32)(size_t)context->runtime->roots[0];
}

// Memory layout for Print section:
// i32 total_print_character_num;
// struct {
//    int type;  // 0 for i32, 1 for f32, 2 for char (i8)
//    union {
//      i32 i;
//      f32 f;
//      char c[4];
//    } data;
//} wasm_buffer_buffer[kWasmPrintBufferSize];
void wasm_set_print_buffer(RuntimeContext *context, Ptr buffer) {
  context->runtime->wasm_print_buffer = buffer;
}

void wasm_print_i32(RuntimeContext *context, i32 value) {
  Ptr buffer = context->runtime->wasm_print_buffer;
  if (buffer == nullptr)
    return;
  i32 total_cnt = ((i32 *)buffer)[0]++;
  i32 print_pos = total_cnt % kWasmPrintBufferSize;
  ((i32 *)buffer)[print_pos * 2 + 1] = 0;  // 0 for i32
  ((i32 *)buffer)[print_pos * 2 + 2] = value;
}

void wasm_print_f32(RuntimeContext *context, f32 value) {
  Ptr buffer = context->runtime->wasm_print_buffer;
  if (buffer == nullptr)
    return;
  i32 total_cnt = ((i32 *)buffer)[0]++;
  i32 print_pos = total_cnt % kWasmPrintBufferSize;
  ((i32 *)buffer)[print_pos * 2 + 1] = 1;  // 1 for f32
  ((f32 *)buffer)[print_pos * 2 + 2] = value;
}

void wasm_print_char(RuntimeContext *context,
                     i8 value0,
                     i8 value1,
                     i8 value2,
                     i8 value3) {
  Ptr buffer = context->runtime->wasm_print_buffer;
  if (buffer == nullptr)
    return;
  i32 total_cnt = ((i32 *)buffer)[0]++;
  i32 print_pos = total_cnt % kWasmPrintBufferSize;
  ((i32 *)buffer)[print_pos * 2 + 1] = 2;  // 2 for char
  ((i8 *)buffer)[print_pos * 8 + 8] = value0;
  ((i8 *)buffer)[print_pos * 8 + 9] = value1;
  ((i8 *)buffer)[print_pos * 8 + 10] = value2;
  ((i8 *)buffer)[print_pos * 8 + 11] = value3;
}

void wasm_set_kernel_parameter_i32(RuntimeContext *context,
                                   int index,
                                   i32 value) {
  *(i32 *)(&context->args[index]) = value;
}

void wasm_set_kernel_parameter_f32(RuntimeContext *context,
                                   int index,
                                   f32 value) {
  *(f32 *)(&context->args[index]) = value;
}
}

#endif
