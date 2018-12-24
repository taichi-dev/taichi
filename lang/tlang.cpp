#include "tlang.h"
#include <taichi/system/timer.h>
#include <taichi/testing.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

TC_TEST("SlowAdapter") {
  /*
  {
    // num_groups, num_inputs, input_group_size, output_group_size
    VV<32, int> input0;
    for (int i = 0; i < 32; i++) {
      input0[i] = i;
    }
    // num_groups, num_inputs, input_group_size, output_group_size
    SlowAdapter<int, 8, 1, 4, 1> adapter;
    adapter.set<0>(input0);
    adapter.shuffle();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 8; j++) {
        TC_CHECK(adapter.get(i)[j] == i + j * 4);
      }
    }
  }
  {
    // num_groups, num_inputs, input_group_size, output_group_size
    SlowAdapter<int, 8, 4, 1, 4> adapter;
    VV<8, int> inputs[4];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 8; j++) {
        inputs[i][j] = i + j * 4;
      }
      adapter.set(i, inputs[i]);
    }
    adapter.shuffle();
    for (int i = 0; i < 32; i++) {
      TC_CHECK(adapter.get(0)[i] == i);
    }
  }
  */
  auto a = set1<int32, 8>(1);
  auto b = set1<int32, 8>(2);
  auto c = a + b;
  for (int i = 0; i < 8; i++) {
    TC_P(c[i]);
  }

  auto p = vvec<int32, 8, 2>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  auto q = vvec<int32, 8, 2>(3);
  auto r = p * q;
  for (int i = 0; i < 16; i++) {
    TC_P(r.d[i / 8][i % 8]);
  }
}
}

TC_NAMESPACE_END
