#include "tlang.h"
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

// if branching is not supported

template <typename T>
void compile(int dim) {
  using namespace Tlang;

  Expr a[dim][dim], b[dim][dim], c[dim][dim];

  int simd_width = 32 / sizeof(float32);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a[i][j] = load(0, simd_width, 0);
      b[i][j] = load(1, simd_width, 0);
    }
  }

  Expr ret;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      auto sum = a[i][0] * b[0][j];
      for (int k = 1; k < dim; k++) {
        sum = sum + a[i][k] * a[k][j];
      }
      ret.store(sum, 2, simd_width, 0);
    }
  }

  CodeGen gen;
  auto code = gen.run(ret);
  std::cout << code << std::endl;
  TC_INFO("Generated Code:\n{}", code);
}

auto tlang = []() { compile<float32>(3); };

TC_REGISTER_TASK(tlang);

TC_NAMESPACE_END
