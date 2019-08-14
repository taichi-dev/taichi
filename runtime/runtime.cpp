// This file will only be compiled with clang into llvm bitcode
// Generated bitcode will likely get inline for performance.

#include <type_traits>
#include <atomic>

#define FORCEINLINE __attribute__((always_inline))

#define STRUCT_FIELD(S, F)                              \
  extern "C" decltype(S::F) S##_get_##F(S *s) {         \
    return s->F;                                        \
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

using float32 = float;
constexpr int taichi_max_num_indices = 4;
constexpr int taichi_max_num_args = 8;

using uint8 = uint8_t;
using Ptr = uint8 *;

using ContextArgType = long long;

extern "C" {

int printf(const char *, ...);

struct PhysicalCoordinates {
  int coordinates[taichi_max_num_indices];
};

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
  uint8 *(*lookup_element)(Ptr, Ptr, int i);
  uint8 *(*from_parent_element)(Ptr, Ptr);
  bool (*is_active)(Ptr, Ptr, int i);
  int (*get_num_elements)(Ptr, Ptr);
  void (*refine_coordinates)(Ptr,
                             PhysicalCoordinates *inp_coord,
                             PhysicalCoordinates *refined_coord,
                             int index);
};

STRUCT_FIELD(StructMeta, element_size)
STRUCT_FIELD(StructMeta, max_num_elements)
STRUCT_FIELD(StructMeta, get_num_elements);
STRUCT_FIELD(StructMeta, lookup_element);
STRUCT_FIELD(StructMeta, from_parent_element);
STRUCT_FIELD(StructMeta, refine_coordinates);

// Specialized Attributes and functions
struct DenseMeta : public StructMeta {
  bool bitmasked;
  bool morton_dim;
};

STRUCT_FIELD(DenseMeta, bitmasked)
STRUCT_FIELD(DenseMeta, morton_dim)

void Dense_activate(Ptr meta, Ptr node, int i) {
}

void *Dense_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}

int Dense_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void *taichi_allocate_aligned(std::size_t size, int alignment);

void *taichi_allocate(std::size_t size) {
  return taichi_allocate_aligned(size, 1);
}

void ___stubs___() {
  printf("");
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

void Runtime_initialize(Runtime **runtime_ptr, int num_snodes) {
  *runtime_ptr = (Runtime *)taichi_allocate(sizeof(Runtime));
  Runtime *runtime = *runtime_ptr;
  printf("Initializing runtime with %d selements\n", num_snodes);
  for (int i = 0; i < num_snodes; i++) {
    runtime->element_lists[i] =
        (ElementList *)taichi_allocate(sizeof(ElementList));
    ElementList_initialize(runtime->element_lists[i]);
  }
  printf("Runtime initialized.\n");
}

// "Element", "component" are different concepts

// ultimately all function calls here will be inlined
void element_listgen(Runtime *runtime, StructMeta *parent, StructMeta *child) {
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->tail;
  auto child_list = runtime->element_lists[child->snode_id];
  for (int i = 0; i < num_parent_elements; i++) {
    auto element = parent_list->elements[i];
    auto ch_component = child->from_parent_element((Ptr)child, element.element);
    int ch_num_elements = child->get_num_elements((Ptr)child, ch_component);
    for (int j = 0; j < ch_num_elements; j++) {
      auto ch_element = child->lookup_element((Ptr)child, element.element, j);
      if (child->is_active((Ptr)child, ch_component, j)) {
        Element elem;
        elem.element = ch_element;
        elem.loop_bounds[0] = 0;
        elem.loop_bounds[1] = child->get_num_elements((Ptr)child, ch_element);
        PhysicalCoordinates refined_coord;
        parent->refine_coordinates((Ptr)parent, &element.pcoord, &refined_coord,
                                   j);
        elem.pcoord = refined_coord;
        child_list->elements[child_list->tail++] = elem;
      }
    }
  }
}
}
