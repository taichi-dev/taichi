#include <atomic>

constexpr int taichi_max_num_args = 4;

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

int test(Context context) {
  printf("");
  return 0;
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
}

// clang-7 -S context.cpp -o context.ll -emit-llvm -std=c++17 -O3
