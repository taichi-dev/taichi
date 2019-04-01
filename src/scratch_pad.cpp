#include "scratch_pad.h"

TLANG_NAMESPACE_BEGIN

TC_TEST("scratch_pad_bounds") {
  Program prog;

  int N = 8;

  Global(x, i32);
  SNode *block;

  layout([&] {
    auto ijk = Indices(0, 1, 2);
    block = &root.dense(ijk, N).dense(ijk, N);
    block->place(x);
  });

  ScratchPad pad(block);

  pad.access({1, 2, -3}, ScratchPad::AccessFlag::read);

  TC_CHECK(pad.bounds[0][0] == 1);
  TC_CHECK(pad.bounds[0][1] == 2);
  TC_CHECK(pad.bounds[0][2] == -3);

  TC_CHECK(pad.bounds[1][0] == 1);
  TC_CHECK(pad.bounds[1][1] == 2);
  TC_CHECK(pad.bounds[1][2] == -3);

  pad.access({4, -2, 5}, ScratchPad::AccessFlag::read);

  TC_CHECK(pad.bounds[0][0] == 1);
  TC_CHECK(pad.bounds[0][1] == -2);
  TC_CHECK(pad.bounds[0][2] == -3);

  TC_CHECK(pad.bounds[1][0] == 4);
  TC_CHECK(pad.bounds[1][1] == 2);
  TC_CHECK(pad.bounds[1][2] == 5);
}

TLANG_NAMESPACE_END
