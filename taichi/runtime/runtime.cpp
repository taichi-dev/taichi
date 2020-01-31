#if !defined(TC_INCLUDED) || !defined(_WIN32)
// This file will only be compiled with clang into llvm bitcode
// Generated bitcode will likely get inline for performance.
// Most function calls here will be inlined

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include "../constants.h"

#if defined(__linux__) && !ARCH_cuda && defined(TI_ARCH_x86_64)
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

template <typename T>
void locked_task(void *lock, const T &func);

template <typename T>
T ifloordiv(T a, T b) {
  auto r = a / b;
  r -= T(a * b < 0 && b * r != a);
  return r;
}

template <typename T>
void Printf(const char *format, T t);

extern "C" {

#if ARCH_cuda
void vprintf(Ptr format, Ptr arg);
#endif
i32 printf(const char *, ...);

#define DEFINE_UNARY_REAL_FUNC(F) \
  f32 F##_f32(f32 x) {            \
    return std::F(x);             \
  }                               \
  f64 F##_f64(f64 x) {            \
    return std::F(x);             \
  }

// sin and cos are already included in llvm intrinsics
DEFINE_UNARY_REAL_FUNC(exp)
DEFINE_UNARY_REAL_FUNC(log)
DEFINE_UNARY_REAL_FUNC(tan)
DEFINE_UNARY_REAL_FUNC(tanh)
DEFINE_UNARY_REAL_FUNC(abs)
DEFINE_UNARY_REAL_FUNC(acos)
DEFINE_UNARY_REAL_FUNC(asin)

int abs_i32(int a) {
  if (a > 0) {
    return a;
  } else {
    return -a;
  }
}

#if ARCH_x86_64

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

int max_i32(int a, int b) {
  return a > b ? a : b;
}

int min_i32(int a, int b) {
  return a < b ? a : b;
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

#include "../context.h"

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
  int snode_id;
  std::size_t element_size;
  int max_num_elements;

  Ptr (*lookup_element)(Ptr, Ptr, int i);

  Ptr (*from_parent_element)(Ptr);

  bool (*is_active)(Ptr, Ptr, int i);

  int (*get_num_elements)(Ptr, Ptr);

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

struct Runtime;

void taichi_assert(Context *context, i32 test, const char *msg);
void taichi_assert_runtime(Runtime *runtime, i32 test, const char *msg);
#define TC_ASSERT_INFO(x, msg) taichi_assert(context, (int)(x), msg)
#define TC_ASSERT(x) TC_ASSERT_INFO(x, #x)

void ___stubs___() {
  printf("");
#if ARCH_cuda
  vprintf(nullptr, nullptr);
#endif
}
}

/*
A simple list data structure
Data are organized in chunks, where each chunk is a piece of virtual memory
*/

bool is_power_of_two(uint32 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

uint32 log2int(uint64 value) {
  uint32 ret = 0;
  value >>= 1;
  while (value) {
    value >>= 1;
    ret += 1;
  }
  return ret;
}

struct ListManager {
  static constexpr std::size_t max_num_chunks = 1024;
  Ptr chunks[max_num_chunks];
  i32 element_size;
  std::size_t max_num_elements_per_chunk;
  i32 log2chunk_num_elements;
  i32 lock;
  i32 num_elements;
  Runtime *runtime;

  ListManager(Runtime *runtime,
              std::size_t element_size,
              std::size_t num_elements_per_chunk)
      : element_size(element_size),
        max_num_elements_per_chunk(num_elements_per_chunk),
        runtime(runtime) {
    taichi_assert_runtime(runtime, is_power_of_two(max_num_elements_per_chunk),
                          "max_num_elements_per_chunk must be POT.");
    lock = 0;
    num_elements = 0;
    log2chunk_num_elements = log2int(num_elements_per_chunk);
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

  void clear() {
    num_elements = 0;
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
      // Printf("i %d\n", i);
      // Printf("chunk %p\n", chunks[i]);
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

using vm_allocator_type = void *(*)(void *, std::size_t, std::size_t);
using RangeForTaskFunc = void(Context *, int i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int i));

constexpr int num_rand_states = 1024 * 32;

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

struct Runtime {
  vm_allocator_type vm_allocator;
  assert_failed_type assert_failed;
  Ptr prog;
  Ptr root;
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

  template <typename T, typename... Args>
  T *create(Args &&... args) {
    auto ptr = (T *)request_allocate_aligned(sizeof(T), 4096);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }
};

struct NodeManager {
  Runtime *runtime;
  i32 lock;
  i32 element_size;
  i32 chunk_num_elements;

  i32 free_list_used;
  ListManager *free_list, *recycled_list, *data_list;

  using list_data_type = i32;

  NodeManager(Runtime *runtime, i32 element_size, i32 chunk_num_elements = -1)
      : runtime(runtime), chunk_num_elements(chunk_num_elements) {
    // 16K elements per chunk
    if (chunk_num_elements == -1)
      chunk_num_elements = 16 * 1024;
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
    free_list_used = 0;

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
STRUCT_FIELD_ARRAY(Runtime, element_lists);
STRUCT_FIELD_ARRAY(Runtime, node_allocators);
STRUCT_FIELD(Runtime, root);
Ptr Runtime_get_temporary_pointer(Runtime *runtime, u64 offset) {
  return runtime->temporaries + offset;
}

STRUCT_FIELD(Runtime, assert_failed);
STRUCT_FIELD(Runtime, mem_req_queue);

#if ARCH_cuda
void __assertfail(const char *message,
                  const char *file,
                  i32 line,
                  const char *function,
                  std::size_t charSize);

void taichi_assert_runtime(Runtime *runtime, i32 test, const char *msg) {
  if (test == 0) {
    __assertfail(msg, "", 1, "", 1);
  }
}
#else
void taichi_assert_runtime(Runtime *runtime, i32 test, const char *msg) {
  if (test == 0) {
    runtime->assert_failed(msg);
  }
}
#endif

void taichi_assert(Context *context, i32 test, const char *msg) {
  taichi_assert_runtime((Runtime *)context->runtime, test, msg);
}

Ptr Runtime::allocate_aligned(std::size_t size, std::size_t alignment) {
  return (Ptr)vm_allocator(prog, size, alignment);
}

Ptr Runtime::allocate(std::size_t size) {
  return allocate_aligned(size, 1);
}

Ptr Runtime::request_allocate_aligned(std::size_t size, std::size_t alignment) {
  auto i = atomic_add_i32(&mem_req_queue->tail, 1);
  taichi_assert_runtime(this, i <= taichi_max_num_mem_requests,
                        "Too many memory allocation requests.");
  auto volatile r = &mem_req_queue->requests[i];
  atomic_exchange_u64((uint64 *)&r->size, size);
  atomic_exchange_u64((uint64 *)&r->alignment, alignment);
  // wait for host to allocate
  while (r->ptr == nullptr)
    ;
  return r->ptr;
}

Ptr Runtime_initialize(Runtime **runtime_ptr,
                       Ptr prog,
                       int num_snodes,
                       uint64_t root_size,
                       int root_id,
                       void *_vm_allocator,
                       bool verbose) {
  // bootstrap
  auto vm_allocator = (vm_allocator_type)_vm_allocator;
  *runtime_ptr = (Runtime *)vm_allocator(prog, sizeof(Runtime), 128);
  Runtime *runtime = *runtime_ptr;
  runtime->vm_allocator = vm_allocator;
  runtime->prog = prog;
  if (verbose)
    printf("Initializing runtime with %d snode(s)...\n", num_snodes);

  // runtime->allocate ready to use
  runtime->mem_req_queue = (MemRequestQueue *)runtime->allocate_aligned(
      sizeof(MemRequestQueue), 4096);

  auto root_ptr = runtime->allocate_aligned(root_size, 4096);

  runtime->temporaries =
      (Ptr)runtime->allocate_aligned(taichi_global_tmp_buffer_size, 1024);

  runtime->rand_states = (RandState *)runtime->allocate_aligned(
      sizeof(RandState) * num_rand_states, 4096);
  for (int i = 0; i < num_rand_states; i++)
    initialize_rand_state(&runtime->rand_states[i], i);

  if (verbose)
    printf("Runtime initialized.\n");
  return (Ptr)root_ptr;
}

void Runtime_initialize2(Runtime *runtime,
                         Ptr root_ptr,
                         int root_id,
                         int num_snodes) {
  // runtime->request_allocate_aligned ready to use

  // initialize the root node element list
  for (int i = 0; i < num_snodes; i++) {
    runtime->element_lists[i] =
        runtime->create<ListManager>(runtime, sizeof(Element), 1024 * 64);
  }
  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = (Ptr)root_ptr;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    elem.pcoord.val[i] = 0;
  }

  runtime->element_lists[root_id]->append(&elem);
}
void Runtime_initialize_thread_pool(Runtime *runtime,
                                    void *thread_pool,
                                    void *parallel_for) {
  runtime->thread_pool = (Ptr)thread_pool;
  runtime->parallel_for = (parallel_for_type)parallel_for;
}

void NodeAllocator_initialize(Runtime *runtime,
                              int snode_id,
                              std::size_t node_size) {
  runtime->node_allocators[snode_id] =
      runtime->create<NodeManager>(runtime, node_size, 1024 * 16);
}

void Runtime_allocate_ambient(Runtime *runtime, int snode_id) {
  runtime->ambient_elements[snode_id] =
      runtime->node_allocators[snode_id]->allocate();
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

int32 grid_dim() {
  return 0;
}

void sync_warp(uint32 mask) {
}

void block_barrier() {
}

int32 warp_active_mask() {
  return 0;
}

void block_memfence() {
}

void threadfence() {
}

// "Element", "component" are different concepts

void clear_list(Runtime *runtime, StructMeta *parent, StructMeta *child) {
  auto child_list = runtime->element_lists[child->snode_id];
  child_list->clear();
}

/*
 * The element list of a SNode, maintains pointers to its instances, and
 * instances' parents' coordinates
 */
void element_listgen(Runtime *runtime, StructMeta *parent, StructMeta *child) {
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->size();
  auto child_list = runtime->element_lists[child->snode_id];
#if ARCH_cuda
  int i_start = block_idx();
  int i_step = grid_dim();
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
    for (int j = element.loop_bounds[0] + j_start; j < element.loop_bounds[1];
         j += j_step) {
      PhysicalCoordinates refined_coord;
      parent->refine_coordinates(&element.pcoord, &refined_coord, j);
      if (parent->is_active((Ptr)parent, element.element, j)) {
        auto ch_element =
            parent->lookup_element((Ptr)parent, element.element, j);
        ch_element = child->from_parent_element((Ptr)ch_element);
        Element elem;
        elem.element = ch_element;
        elem.loop_bounds[0] = 0;
        elem.loop_bounds[1] = child->get_num_elements((Ptr)child, ch_element);
        elem.pcoord = refined_coord;
        child_list->append(&elem);
      }
    }
  }
}

using BlockTask = void(Context *, Element *, int, int);

struct block_task_helper_context {
  Context *context;
  BlockTask *task;
  ListManager *list;
  int element_size;
  int element_split;
};

void block_helper(void *ctx_, int i) {
  auto ctx = (block_task_helper_context *)(ctx_);
  int element_id = i / ctx->element_split;
  int part_size = ctx->element_size / ctx->element_split;
  int part_id = i % ctx->element_split;
  // printf("%d %d %d\n", element_id, part_size, part_id);
  auto &e = ctx->list->get<Element>(element_id);
  int lower = e.loop_bounds[0] + part_id * part_size;
  int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
  upper = std::min(upper, e.loop_bounds[1]);
  if (lower < upper) {
    (*ctx->task)(ctx->context, &ctx->list->get<Element>(element_id), lower,
                 upper);
  }
}

void for_each_block(Context *context,
                    int snode_id,
                    int element_size,
                    int element_split,
                    BlockTask *task,
                    int num_threads) {
  auto list = ((Runtime *)context->runtime)->element_lists[snode_id];
  auto list_tail = list->size();
#if ARCH_cuda
  int i = block_idx();
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
  block_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.list = list;
  ctx.element_size = element_size;
  ctx.element_split = element_split;
  // printf("size %d spilt %d tail %d\n", ctx.element_size, ctx.element_split,
  // list_tail);
  auto runtime = (Runtime *)context->runtime;
  runtime->parallel_for(runtime->thread_pool, list_tail * element_split,
                        num_threads, &ctx, block_helper);
#endif
}

struct range_task_helper_context {
  Context *context;
  RangeForTaskFunc *task;
  int begin;
  int end;
  int block_size;
  int step;
};

void parallel_range_for_task(void *range_context, int task_id) {
  auto ctx = *(range_task_helper_context *)range_context;
  if (ctx.step == 1) {
    int block_start = ctx.begin + task_id * ctx.block_size;
    int block_end = std::min(block_start + ctx.block_size, ctx.end);
    for (int i = block_start; i < block_end; i++) {
      ctx.task(ctx.context, i);
    }
  } else if (ctx.step == -1) {
    int block_start = ctx.end - task_id * ctx.block_size;
    int block_end = std::max(ctx.begin, block_start * ctx.block_size);
    for (int i = block_start - 1; i >= block_end; i--) {
      ctx.task(ctx.context, i);
    }
  }
}

void cpu_parallel_range_for(Context *context,
                            int num_threads,
                            int begin,
                            int end,
                            int step,
                            int block_dim,
                            RangeForTaskFunc *task) {
  range_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.begin = begin;
  ctx.end = end;
  ctx.block_size = block_dim;
  ctx.step = step;
  if (step != 1 && step != -1) {
    printf("step must not be %d\n", step);
    exit(-1);
  }
  auto runtime = (Runtime *)context->runtime;
  runtime->parallel_for(runtime->thread_pool,
                        (end - begin + block_dim - 1) / block_dim, num_threads,
                        &ctx, parallel_range_for_task);
}

void gpu_parallel_range_for(Context *context,
                            int begin,
                            int end,
                            RangeForTaskFunc *func) {
  int idx = thread_idx() + block_dim() * block_idx() + begin;
  while (idx < end) {
    func(context, idx);
    idx += block_dim() * grid_dim();
  }
}

i32 linear_thread_idx() {
  return block_idx() * block_dim() + thread_idx();
}

#include "node_dense.h"
#include "node_dynamic.h"
#include "node_pointer.h"
#include "node_root.h"

#include "internal_function.h"

void ListManager::touch_chunk(int chunk_id) {
  if (!chunks[chunk_id]) {
    // Printf("chunkid %d\n", chunk_id);
    locked_task(&lock, [&] {
      // may have been allocated during lock contention
      if (!chunks[chunk_id]) {
        // Printf("Allocating chunk %d\n", chunk_id);
        chunks[chunk_id] = runtime->request_allocate_aligned(
            max_num_elements_per_chunk * element_size, 4096);
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

void node_gc(Runtime *runtime, int snode_id) {
  runtime->node_allocators[snode_id]->gc_serial();
}
}

#if ARCH_cuda

extern "C" {

u32 cuda_rand_u32(Context *context) {
  auto state = &((Runtime *)context->runtime)
                    ->rand_states[linear_thread_idx() % num_rand_states];
  u32 ret;
  auto lock = (Ptr)&state->lock;

  bool done = false;
  // TODO: whether this leads to a deadlock or not depends on how nvcc schedules
  // the instructions...
  while (!done) {
    if (atomic_exchange_i32((i32 *)lock, 1) == 1) {
      auto &x = state->x;
      auto &y = state->y;
      auto &z = state->z;
      auto &w = state->w;
      auto t = x ^ (x << 11);
      x = y;
      y = z;
      z = w;
      w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
      ret = w;
      done = true;
      mutex_unlock_i32(lock);
    }
  }
  return ret;
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

template <typename T>
void Printf(const char *format, T t) {
#if ARCH_cuda
  vprintf((Ptr)format, (Ptr)&t);
#else
  printf(format, t);
#endif
}

#include "locked_task.h"

#endif
