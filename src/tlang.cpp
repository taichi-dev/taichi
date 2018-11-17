#include "tlang.h"
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

auto test_tlang = []() {
  using namespace T;
  Expr a, b, c, d;

  auto ret = (a * b + c) / d;
  auto ret2 = ret - ret;

  CodeGen gen;
  auto code = gen.run(ret2);
  std::cout << code << std::endl;
  TC_INFO("Generated Code:\n{}", code);
};

TC_REGISTER_TASK(test_tlang);

TC_NAMESPACE_END
