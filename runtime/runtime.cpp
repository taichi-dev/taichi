// This file will only be compiled with clang into llvm bitcode
// Generated bitcode will likely get inline for performance.

#include <atomic>

#define FORCEINLINE __attribute__((always_inline))

constexpr int taichi_max_num_indices = 4;
constexpr int taichi_max_num_args = 8;

using ContextArgType = long long;

struct Context {
  void *buffer;
  ContextArgType args[taichi_max_num_args];
  void *leaves;
  int num_leaves;
  void *cpu_profiler;
};

extern "C" {

ContextArgType context_get_arg(Context *context, int arg_id) {
  return context->args[arg_id];
}

void *context_get_buffer(Context *context) {
  return context->buffer;
}

int printf(const char *, ...);

using float32 = float;

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

struct NodeRef {
  int indices[taichi_max_num_indices];
  int loop_bounds[2];
  void *node;
};

int *noderef_get_index_ptr(NodeRef *noderef, int i) {
  return &noderef->indices[i];
}

int *noderef_get_loop_bound_ptr(NodeRef *noderef, int i) {
  return &noderef->loop_bounds[i];
}

void **noderef_get_node_ptr_ptr(NodeRef *noderef) {
  return &noderef->node;
}

#define STRUCT_FIELD(S, F)                              \
  extern "C" decltype(S::F) S##_get_##F(S *s) {         \
    return s->F;                                        \
  }                                                     \
  extern "C" void S##_set_##F(S *s, decltype(S::F) f) { \
    s->F = f;                                           \
  }

/*
#define STRUCT_FUNCTION(S, F)
extern "C" decltype()
 */

// These structures are accessible by both the LLVM backend and this C++ runtime
// file here (for building complex runtime functions in C++)

struct DenseStruct {
  void *node;
  bool bitmasked;
  int morton_dim;
  std::size_t element_size;
  int forking_factor;  // n
};

STRUCT_FIELD(DenseStruct, node)
STRUCT_FIELD(DenseStruct, bitmasked)
STRUCT_FIELD(DenseStruct, morton_dim)
STRUCT_FIELD(DenseStruct, element_size)
STRUCT_FIELD(DenseStruct, forking_factor)

void DenseStruct_activate(DenseStruct *s, int i) {
}

void *DenseStruct_lookup(DenseStruct *s, int i) {
  return (char *)s->node + s->element_size * i;
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

struct Node {
  void *node;
  int coordinates[taichi_max_num_indices];
};

STRUCT_FIELD(Node, node);
// STRUCT_FIELD(Node, coordinates);

struct NodeList {
  Node *list;
  int tail;
};

void NodeList_initialize(NodeList *node_list) {
  node_list->list = (Node *)taichi_allocate(1024 * 1024 * 1024);
  node_list->tail = 0;
}

void NodeList_insert(NodeList *node_list, Node *node) {
  node_list->list[node_list->tail] = *node;
  node_list->tail++;
}

void NodeList_clear(NodeList *node_list) {
  node_list->tail = 0;
}

// Is "runtime" a correct name, even if it is created after the data layout is
// materialized?
struct Runtime {
  NodeList *lists[1024];
};

void Runtime_initialize(Runtime *runtime, int num_snodes) {
  for (int i = 0; i < num_snodes; i++) {
    runtime->lists[i] = (NodeList *)taichi_allocate(sizeof(NodeList));
    NodeList_initialize(runtime->lists[i]);
  }
}

/*
 *
void unitied_allocator_create();

void unified_allocator_free();

void *create_noderef_vector() {
  return new std::vector<NodeRef>;
}

void destory_noderef_vector(void *vec) {
  auto ptr = (std::vector<NodeRef> *)vec;
  delete ptr;
}

int noderef_vector_size(void *vec) {
  auto ptr = (std::vector<NodeRef> *)vec;
  return ptr->size();
}

NodeRef *get_noderef_vector_elem(void *vec, int i) {
  auto ptr = (std::vector<NodeRef> *)vec;
  return &(*ptr)[i];
}
*/
}
