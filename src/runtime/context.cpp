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

ContextArgType *context_get_args(Context *context, int arg_id) {
  return &context->args[arg_id];
}

void *context_get_buffer(Context *context) {
  return context->buffer;
}

int test(Context context) {
  return *context_get_args(&context, 0);
}
}

// clang-7 -S context.cpp -o context.ll -emit-llvm -std=c++17
