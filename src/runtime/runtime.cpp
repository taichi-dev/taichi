#include <atomic>
// #include <vector>

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

void ___test___() {
  printf("");
}

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
// file here (for building complex runtime functions)

struct DenseStruct {
  void *node;
  bool bitmasked;
  int morton_dim;
  int forking_factor;  // n

  void activate(int i) {

  }
};

STRUCT_FIELD(DenseStruct, node);
STRUCT_FIELD(DenseStruct, bitmasked);
STRUCT_FIELD(DenseStruct, morton_dim);
// STRUCT_FUNCTION(DenseStruct, activate, int);

void struct_dense_activate(void *) {
}

/*
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
