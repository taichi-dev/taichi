#pragma once
#include "common.h"
#include <string>

namespace taichi {
namespace Tlang {

class CPUProfiler;

struct Context {
  using Buffer = void *;
  Buffer buffers[1];

  void *leaves;
  int num_leaves;
  CPUProfiler *cpu_profiler;

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

}  // namespace Tlang

};  // namespace taichi
