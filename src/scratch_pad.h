#include <taichi/common/testing.h>
#include "tlang.h"

TLANG_NAMESPACE_BEGIN

class ScratchPad {
 public:
  enum AccessFlag : int { read = 1 << 1, write = 1 << 2 };

  SNode *snode;

  std::vector<int> bounds[2];
  std::vector<int> pad_size;
  std::vector<int> block_size;
  int dim;
  bool empty;

  std::vector<AccessFlag> flags;

  ScratchPad(SNode *snode) {
    dim = snode->num_active_indices;
    bounds[0].resize(dim);
    bounds[1].resize(dim);
    pad_size.resize(dim);

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
      pad_size[i] = bounds[1][i] - bounds[0][i];
    }
  }

  void compile() {
    int size = 1;
    for (int i = 0; i < dim; i++) {
      size *= pad_size[i];
    }
    flags.resize(size);

    for (int i = 0; i < dim; i++) {
      block_size[i] =
          1 << snode->extractors[snode->physical_index_position[i]].num_bits;
    }

    TC_ASSERT(dim == 1);
    for (int i = 0; i < pad_size[0]; i++) {
    }
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

  void CSE() {
  }

  void print() {

  }


};

TLANG_NAMESPACE_END
