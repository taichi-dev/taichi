#include "tlang.h"
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

// if branching is not supported

using namespace Tlang;

auto test_tlang = []() {
  Address addr;
  addr.stream_id = 0;
  addr.coeff_i = 1;
  Expr a = load(addr);
  addr.stream_id = 1;
  Expr b = load(addr);
  auto c = a + b;
  Expr ret;
  addr.stream_id = 2;
  ret.store(c, addr);
  CodeGen cg;
  auto func = cg.get(ret, 8);

  float32 x[16], y[16], z[16];
  for (int i = 0; i < 16; i++) {
    x[i] = i;
    y[i] = -2 * i;
  }
  func(x, y, z, 16);
  for (int i = 0; i < 16; i++) {
    TC_P(i);
    TC_P(z[i]);
  }
};
TC_REGISTER_TASK(test_tlang);

TC_NAMESPACE_END
