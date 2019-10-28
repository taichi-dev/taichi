// This file will only be compiled with clang into llvm bitcode
// Generated bitcode will likely get inline for performance.

#include <type_traits>
#include <atomic>
#include <cmath>

#define FORCEINLINE __attribute__((always_inline))

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
using int32 = int32_t;
using int64 = int64_t;
using float32 = float;
using float64 = double;

using i8 = int8;
using i32 = int32;
using f32 = float32;
using f64 = float64;

constexpr int taichi_max_num_indices = 4;
constexpr int taichi_max_num_args = 8;

using uint8 = uint8_t;
using Ptr = uint8 *;

using ContextArgType = long long;

extern "C" {

void vprintf(Ptr format, Ptr arg);
i32 printf(const char *, ...);

#define DEFINE_UNARY_REAL_FUNC(F) \
  float F##_f32(float x) {        \
    return std::F(x);             \
  }                               \
  double F##_f64(double x) {      \
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
  void *leaves;
  int num_leaves;
  void *cpu_profiler;
  Ptr runtime;
};

STRUCT_FIELD_ARRAY(Context, args);
STRUCT_FIELD(Context, runtime);
STRUCT_FIELD(Context, buffer);

float32 atomic_add_cpu_f32(volatile float32 *dest, float32 inc) {
  float32 old_val;
  float32 new_val;
  do {
    old_val = *dest;
    new_val = old_val + inc;
#if defined(__clang__)
  } while (!__atomic_compare_exchange(dest, &old_val, &new_val, true,
                                      std::memory_order::memory_order_seq_cst,
                                      std::memory_order::memory_order_seq_cst));
#else
  } while (!__atomic_compare_exchange((float32 *)dest, &old_val, &new_val, true,
                                      std::memory_order::memory_order_seq_cst,
                                      std::memory_order::memory_order_seq_cst));
#endif
  return old_val;
}
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
};

STRUCT_FIELD(StructMeta, snode_id)
STRUCT_FIELD(StructMeta, element_size)
STRUCT_FIELD(StructMeta, max_num_elements)
STRUCT_FIELD(StructMeta, get_num_elements);
STRUCT_FIELD(StructMeta, lookup_element);
STRUCT_FIELD(StructMeta, from_parent_element);
STRUCT_FIELD(StructMeta, refine_coordinates);
STRUCT_FIELD(StructMeta, is_active);

// Specialized Attributes and functions
struct DenseMeta : public StructMeta {
  bool bitmasked;
  int morton_dim;
};

STRUCT_FIELD(DenseMeta, bitmasked)
STRUCT_FIELD(DenseMeta, morton_dim)

void Dense_activate(Ptr meta, Ptr node, int i) {
}

bool Dense_is_active(Ptr meta, Ptr node, int i) {
  return true;
}

void *Dense_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}

int Dense_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

struct RootMeta : public StructMeta {
  int tag;
};

STRUCT_FIELD(RootMeta, tag);

void Root_activate(Ptr meta, Ptr node, int i) {
}

bool Root_is_active(Ptr meta, Ptr node, int i) {
  return true;
}

void *Root_lookup_element(Ptr meta, Ptr node, int i) {
  // only one element
  return node;
}

int Root_get_num_elements(Ptr meta, Ptr node) {
  return 1;
}

void *taichi_allocate_aligned(std::size_t size, int alignment);

void *taichi_allocate(std::size_t size) {
  return taichi_allocate_aligned(size, 1);
}

void ___stubs___() {
  printf("");
  vprintf(nullptr, nullptr);
  taichi_allocate(1);
  taichi_allocate_aligned(1, 1);
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

void ElementList_initialize(ElementList *element_list) {
  element_list->elements = (Element *)taichi_allocate(1024 * 1024 * 1024);
  element_list->tail = 0;
}

void ElementList_insert(ElementList *element_list, Element *element) {
  element_list->elements[element_list->tail] = *element;
  element_list->tail++;
}

void ElementList_clear(ElementList *element_list) {
  element_list->tail = 0;
}

// Is "runtime" a correct name, even if it is created after the data layout is
// materialized?
struct Runtime {
  ElementList *element_lists[1024];
};

STRUCT_FIELD_ARRAY(Runtime, element_lists);

Ptr Runtime_initialize(Runtime **runtime_ptr,
                       int num_snodes,
                       uint64_t root_size,
                       int root_id) {
  *runtime_ptr = (Runtime *)taichi_allocate(sizeof(Runtime));
  Runtime *runtime = *runtime_ptr;
  printf("Initializing runtime with %d elements\n", num_snodes);
  for (int i = 0; i < num_snodes; i++) {
    runtime->element_lists[i] =
        (ElementList *)taichi_allocate(sizeof(ElementList));
    ElementList_initialize(runtime->element_lists[i]);
  }
  // Assuming num_snodes - 1 is the root
  auto root_ptr = taichi_allocate_aligned(root_size, 4096);
  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = (Ptr)root_ptr;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    elem.pcoord.val[i] = 0;
  }
  ElementList_insert(runtime->element_lists[root_id], &elem);
  printf("Runtime initialized.\n");
  return (Ptr)root_ptr;
}

// "Element", "component" are different concepts

// ultimately all function calls here will be inlined
void element_listgen(Runtime *runtime, StructMeta *parent, StructMeta *child) {
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->tail;
  // printf("num_parent_elements %d\n", num_parent_elements);
  auto child_list = runtime->element_lists[child->snode_id];
  child_list->head = 0;
  for (int i = 0; i < num_parent_elements; i++) {
    auto element = parent_list->elements[i];
    auto ch_component = child->from_parent_element(element.element);
    int ch_num_elements = child->get_num_elements((Ptr)child, ch_component);
    // printf("ch_num_parent_elements %d\n", ch_num_elements);
    for (int j = 0; j < ch_num_elements; j++) {
      auto ch_element = child->lookup_element((Ptr)child, element.element, j);
      // printf("j %d\n", j);
      if (child->is_active((Ptr)child, ch_component, j)) {
        // printf("j! %d\n", j);
        Element elem;
        elem.element = ch_element;
        elem.loop_bounds[0] = 0;
        elem.loop_bounds[1] = child->get_num_elements((Ptr)child, ch_element);
        PhysicalCoordinates refined_coord;
        child->refine_coordinates(&element.pcoord, &refined_coord, j);
        /*
        printf("snode id %d\n", child->snode_id);
        for (int k = 0; k < taichi_max_num_indices; k++) {
          printf("   %d\n", refined_coord.val[k]);
        }
        */
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

void block_barrier() {
}

#if ARCH_cuda
void for_each_block(Context *context,
                    int snode_id,
                    void (*task)(Context *, Element *)) {
  /*
  auto list = ((Runtime *)context->runtime)->element_lists[snode_id];
  block_barrier();
  if (thread_idx() == 0) {
    atomic_add(&list->head, 1);
  }
  block_barrier();
  for (int i = 0; i < list->tail; i++) {
    task(context, &list->elements[i]);
  }
  */
  auto list = ((Runtime *)context->runtime)->element_lists[snode_id];
  for (int i = 0; i < list->tail; i++) {
    task(context, &list->elements[i]);
  }
}
#else
void for_each_block(Context *context,
                    int snode_id,
                    void (*task)(Context *, Element *)) {
  auto list = ((Runtime *)context->runtime)->element_lists[snode_id];
  for (int i = 0; i < list->tail; i++) {
    task(context, &list->elements[i]);
  }
}
#endif

int ti_cuda_tid_x() {
  return 0;
}

void cuda_add(float *a, float *b, float *c) {
  auto i = ti_cuda_tid_x();
  c[i] = a[i] + b[i];
}
}
