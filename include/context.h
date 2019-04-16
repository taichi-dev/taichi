#pragma once
#include "common.h"
#include <string>

namespace taichi {
namespace Tlang {

struct Context {
  using Buffer = void *;
  Buffer buffers[1];

  void *leaves;
  int num_leaves;

  Context() {
    leaves = 0;
    num_leaves = 0;
    for (int i = 0; i < 1; i++)
      buffers[i] = nullptr;
  }

  Context(void *x) : Context() {
    buffers[0] = x;
  }
};

/*
#if defined(TC_GPU)
__device__ __host__
#endif
    void *
    allocate(std::size_t size);
*/
}  // namespace Tlang

};  // namespace taichi
