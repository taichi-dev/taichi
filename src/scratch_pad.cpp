#include <taichi/common/testing.h>
#include "tlang.h"

TLANG_NAMESPACE_BEGIN

class ScratchPad {
 public:
  enum AccessFlag : int { read = 1 << 1, write = 1 << 2 };

  SNode *snode;

  std::vector<int> bounds[2];
  int dim;
  bool empty;

  ScratchPad(SNode *snode) {
    dim = snode->num_active_indices;
    bounds[0].resize(dim);
    bounds[1].resize(dim);

    std::fill(bounds[0].begin(), bounds[0].end(),
              std::numeric_limits<int>::max());
    std::fill(bounds[1].begin(), bounds[1].end(),
              std::numeric_limits<int>::min());
    empty = false;
  }

  void access(const std::vector<int> &indices, AccessFlag flags) {
    empty = true;
    TC_ASSERT(indices.size() == dim);
    for (int i = 0; i < dim; i++) {
      bounds[0][i] = std::min(bounds[0][i], indices[i]);
      bounds[1][i] = std::max(bounds[1][i], indices[i]);
    }
  }

  void compile() {
  }

  void codegen_cpu() {
  }
};

class ScratchPads {
 public:
  std::map<SNode *, ScratchPad> pads;

  using AccessFlag = ScratchPad::AccessFlag;

  void access(const SNode *snode,
              const std::vector<int> &indices,
              AccessFlag flags) {
  }

  void compile() {
    for (auto &pad : pads) {
      pad.second.compile();
    }
  }
};

TC_TEST("scratch_pad") {
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
