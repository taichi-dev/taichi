#include "tlang.h"
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

// if branching is not supported

using namespace Tlang;

auto test_tlang = []() {
  Expr a = load(0, 1, 0);
  Expr b = load(1, 1, 0);
  auto c = a + b;
  Expr ret;
  ret.store(c, 2, 1, 0);
  CodeGen cg;
  auto func = cg.get(ret);

  float32 x[16], y[16], z[16];
  for (int i = 0; i < 16; i++) {
    x[i] = i;
    y[i] = -2 * i;
  }
  func(x, y, z, 16);
  for (int i = 0; i < 16; i++) {
    TC_P(z[i]);
  }
};
TC_REGISTER_TASK(test_tlang);

TC_NAMESPACE_END
