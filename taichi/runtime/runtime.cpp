#if !defined(TC_INCLUDED) || !defined(_WIN32)
// This file will only be compiled with clang into llvm bitcode
// Generated bitcode will likely get inline for performance.

#include <atomic>
#if !ARCH_cuda
#include <mutex>
#endif
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

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
using i32 = int32;
using i64 = int64;
using u8 = uint8;
using u32 = uint32;
using u64 = uint64;
using f32 = float32;
using f64 = float64;

// TODO: DRY. merge this with taichi_core
constexpr int taichi_max_num_indices = 4;
constexpr int taichi_max_num_args = 8;
constexpr int taichi_max_num_snodes = 1024;

using uint8 = uint8_t;
using Ptr = uint8 *;

using ContextArgType = long long;

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
  static std::mutex mut;
  std::lock_guard _(mut);
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

f32 __nv_sgnf(f32 x) {
  return sgn_f32(x);
}

f64 __nv_sgn(f64 x) {
  return sgn_f64(x);
}

struct PhysicalCoordinates {
  int val[taichi_max_num_indices];
};

STRUCT_FIELD_ARRAY(PhysicalCoordinates, val);

struct Context {
  void *buffer;
  ContextArgType args[taichi_max_num_args];
  int32 extra_args[taichi_max_num_args][taichi_max_num_indices];
  void *leaves;
  int num_leaves;
  void *cpu_profiler;
  Ptr runtime;
};

STRUCT_FIELD_ARRAY(Context, args);
STRUCT_FIELD(Context, runtime);
STRUCT_FIELD(Context, buffer);

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
void *allocate_aligned(Runtime *, std::size_t size, int alignment);

void *allocate(Runtime *runtime, std::size_t size) {
  return allocate_aligned(runtime, size, 1);
}

void ___stubs___() {
  printf("");
#if ARCH_cuda
  vprintf(nullptr, nullptr);
#endif
}

struct Element {
  uint8 *element;
  int loop_bounds[2];
  PhysicalCoordinates pcoord;
};

STRUCT_FIELD(Element, element);
STRUCT_FIELD(Element, pcoord);
STRUCT_FIELD_ARRAY(Element, loop_bounds);

struct ElementList {
  Element *elements;
  int head;
  int tail;
};

void ElementList_initialize(Runtime *runtime, ElementList *element_list) {
#if defined(_WIN32)
  auto list_size = 32 * 1024 * 1024;
#else
  auto list_size = 1024 * 1024 * 1024;
#endif
  element_list->elements = (Element *)allocate(runtime, list_size);
  element_list->tail = 0;
}

void ElementList_insert(ElementList *element_list, Element *element) {
  element_list->elements[element_list->tail] = *element;
  element_list->tail++;
}

void ElementList_clear(ElementList *element_list) {
  element_list->tail = 0;
}

struct NodeAllocator {
  Ptr pool;
  std::size_t node_size;
  int tail;
};

void NodeAllocator_initialize(Runtime *runtime,
                              NodeAllocator *node_allocator,
                              std::size_t node_size) {
  node_allocator->pool =
      (Ptr)allocate_aligned(runtime, 1024 * 1024 * 1024, 4096);
  node_allocator->node_size = node_size;
  node_allocator->tail = 0;
}

Ptr NodeAllocator_allocate(NodeAllocator *node_allocator) {
  int p = atomic_add_i32(&node_allocator->tail, 1);
  return node_allocator->pool + node_allocator->node_size * p;
}

using vm_allocator_type = void *(*)(std::size_t, int);
using CPUTaskFunc = void(Context *, int i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int i));

constexpr int max_rand_states = 1024 * 1024;

struct RandState {
  u32 x;
  u32 y;
  u32 z;
  u32 w;
};

void initialize_rand_state(RandState *state, u32 i) {
  state->x = 123456789 * i * 1000000007;
  state->y = 362436069;
  state->z = 521288629;
  state->w = 88675123;
}

// Is "runtime" a correct name, even if it is created after the data layout is
// materialized?
struct Runtime {
  vm_allocator_type vm_allocator;
  Ptr thread_pool;
  parallel_for_type parallel_for;
  ElementList *element_lists[taichi_max_num_snodes];
  NodeAllocator *node_allocators[taichi_max_num_snodes];
  Ptr ambient_elements[taichi_max_num_snodes];
  Ptr temporaries;
  RandState *rand_states;
};

STRUCT_FIELD_ARRAY(Runtime, element_lists);
STRUCT_FIELD_ARRAY(Runtime, node_allocators);
STRUCT_FIELD(Runtime, temporaries);

void *allocate_aligned(Runtime *runtime, std::size_t size, int alignment) {
  return runtime->vm_allocator(size, alignment);
}

Ptr Runtime_initialize(Runtime **runtime_ptr,
                       int num_snodes,
                       uint64_t root_size,
                       int root_id,
                       void *_vm_allocator) {
  auto vm_allocator = (vm_allocator_type)_vm_allocator;
  *runtime_ptr = (Runtime *)vm_allocator(sizeof(Runtime), 128);
  Runtime *runtime = *runtime_ptr;
  runtime->vm_allocator = vm_allocator;
  printf("Initializing runtime with %d elements\n", num_snodes);
  for (int i = 0; i < num_snodes; i++) {
    runtime->element_lists[i] =
        (ElementList *)allocate(runtime, sizeof(ElementList));
    ElementList_initialize(runtime, runtime->element_lists[i]);

    runtime->node_allocators[i] =
        (NodeAllocator *)allocate(runtime, sizeof(NodeAllocator));
  }
  // Assuming num_snodes - 1 is the root
  auto root_ptr = allocate_aligned(runtime, root_size, 4096);

  // The same "1048576" is also used in offload.cpp
  // TODO: DRY
  runtime->temporaries = (Ptr)allocate_aligned(runtime, 1048576, 1024);

  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = (Ptr)root_ptr;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    elem.pcoord.val[i] = 0;
  }
  ElementList_insert(runtime->element_lists[root_id], &elem);
  runtime->rand_states = (RandState *)allocate_aligned(
      runtime, sizeof(RandState) * max_rand_states, 4096);
  for (int i = 0; i < max_rand_states; i++)
    initialize_rand_state(&runtime->rand_states[i], i);
  printf("Runtime initialized.\n");
  return (Ptr)root_ptr;
}

void Runtime_initialize_thread_pool(Runtime *runtime,
                                    void *thread_pool,
                                    void *parallel_for) {
  runtime->thread_pool = (Ptr)thread_pool;
  runtime->parallel_for = (parallel_for_type)parallel_for;
}

void Runtime_allocate_ambient(Runtime *runtime, int snode_id) {
  runtime->ambient_elements[snode_id] =
      NodeAllocator_allocate(runtime->node_allocators[snode_id]);
}

// "Element", "component" are different concepts

// ultimately all function calls here will be inlined
void element_listgen(Runtime *runtime, StructMeta *parent, StructMeta *child) {
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->tail;
  auto child_list = runtime->element_lists[child->snode_id];
  child_list->head = 0;
  child_list->tail = 0;
  for (int i = 0; i < num_parent_elements; i++) {
    auto element = parent_list->elements[i];
    auto ch_component = child->from_parent_element(element.element);
    int ch_num_elements = child->get_num_elements((Ptr)child, ch_component);
    for (int j = 0; j < ch_num_elements; j++) {
      if (child->is_active((Ptr)child, ch_component, j)) {
        auto ch_element = child->lookup_element((Ptr)child, element.element, j);
        Element elem;
        elem.element = ch_element;
        elem.loop_bounds[0] = 0;
        elem.loop_bounds[1] = child->get_num_elements((Ptr)child, ch_element);
        PhysicalCoordinates refined_coord;
        child->refine_coordinates(&element.pcoord, &refined_coord, j);
        elem.pcoord = refined_coord;
        ElementList_insert(child_list, &elem);
      }
    }
  }
}

int32 thread_idx() {
  return 0;
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

using BlockTask = void(Context *, Element *, int, int);

struct block_task_helper_context {
  Context *context;
  BlockTask *task;
  Element *list;
  int element_size;
  int element_split;
};

void block_helper(void *ctx_, int i) {
  auto ctx = (block_task_helper_context *)(ctx_);
  int element_id = i / ctx->element_split;
  int part_size = ctx->element_size / ctx->element_split;
  int part_id = i % ctx->element_split;
  (*ctx->task)(ctx->context, &ctx->list[element_id], part_id * part_size,
               (part_id + 1) * part_size);
}

void for_each_block(Context *context,
                    int snode_id,
                    int element_size,
                    int element_split,
                    BlockTask *task,
                    int num_threads) {
  auto list = ((Runtime *)context->runtime)->element_lists[snode_id];
  auto list_tail = list->tail;
#if ARCH_cuda
  int i = block_idx();
  const auto part_size = element_size / element_split;
  while (true) {
    int element_id = i / element_split;
    if (element_id >= list_tail)
      break;
    auto part_id = i % element_split;
    auto lower = part_size * part_id;
    auto upper = part_size * (part_id + 1);
    task(context, &list->elements[element_id], lower, upper);
    i += grid_dim();
  }
#else
  block_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.list = list->elements;
  ctx.element_size = element_size;
  ctx.element_split = element_split;
  auto runtime = (Runtime *)context->runtime;
  runtime->parallel_for(runtime->thread_pool, list_tail * element_split,
                        num_threads, &ctx, block_helper);
#endif
}

struct range_task_helper_context {
  Context *context;
  CPUTaskFunc *task;
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
                            CPUTaskFunc *task) {
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

i32 linear_thread_id() {
  return block_idx() * block_dim() + thread_idx();
}

#if ARCH_cuda

u32 cuda_rand_u32(Context *context) {
  auto state = &((Runtime *)context->runtime)
                    ->rand_states[linear_thread_id() % max_rand_states];
  auto &x = state->x;
  auto &y = state->y;
  auto &z = state->z;
  auto &w = state->w;
  auto t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
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

#endif

#include "node_dense.h"
#include "node_dynamic.h"
#include "node_pointer.h"
#include "node_root.h"
}
#endif
