#ifdef TI_INSIDE_VULKAN_CODEGEN

#include "taichi/util/macros.h"

#else

#define STR(...) __VA_ARGS__

#define inout

// GLSL builtin stubs
int floatBitsToInt(float f) {
  return *reinterpret_cast<int *>(&f);
}

int intBitsToFloat(float f) {
  return *reinterpret_cast<int *>(&f);
}

int atomicCompSwap(int &mem, int compare, int data) {
  const int old = mem;
  if (mem == compare) {
    mem = data;
  }
  return old;
}

#endif  // TI_INSIDE_VULKAN_CODEGEN
