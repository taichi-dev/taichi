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

  ScratchPad() = default;

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
      bounds[1][i] = std::max(bounds[1][i], indices[i] + 1);
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

  std::string name() {
    return snode->node_type_name + "_scratch_pad";
  }

  int linear_size() {
    int s = 1;
    for (int i = 0; i < dim; i++) {
      s *= pad_size[i];
    }
    return s;
  }

  std::string initialize() {
    return fmt::format("{} {}[{}];", snode->node_type_name, name(),
                       linear_size());
  }

  std::string finalize() {
  }
};

inline int div_floor(int a, int b) {
  return a >= 0 ? a / b : (a - b + 1) / b;
}

class ScratchPads {
 public:
  std::map<SNode *, ScratchPad> pads;

  using AccessFlag = ScratchPad::AccessFlag;

  void access(SNode *snode, const std::vector<int> &indices, AccessFlag flags) {
    if (pads.find(snode) == pads.end()) {
      pads.emplace(std::piecewise_construct, std::forward_as_tuple(snode),
                   std::forward_as_tuple(snode));
    }
    pads.find(snode)->second.access(indices, flags);
    if (snode->parent->type != SNodeType::root) {
      auto parent_indices = indices;
      for (int i = 0; i < snode->parent->num_active_indices; i++) {
        int block_dim =
            snode->parent->extractors[snode->parent->physical_index_position[i]]
                .dimension;
        parent_indices[i] = div_floor(parent_indices[i], block_dim);
      }
      access(snode->parent, parent_indices, flags);
    }
  }

  void compile() {
    for (auto &pad : pads) {
      pad.second.compile();
    }
  }

  void CSE() {
  }

  void emit_gather_code_cpu() {
  }

  void emit_gather_code_gpu() {
  }

  void generate_address_code(SNode *snode, const std::vector<int> &indices) {
    if (pads.find(snode) != pads.end()) {
      auto &pad = pads[snode];
      int offset = 0;
      // for (int i = pad.dim - 1; i >= 0; i--) {
      for (int i = 0; i < pad.dim; i++) {
        offset = offset + (indices[i] - pad.bounds[0][i]);
        if (i > 0)
          offset = offset * pad.pad_size[i - 1];
      }
    } else if (pads.find(snode->parent) != pads.end()) {
    } else {
      TC_NOT_IMPLEMENTED
    }
  }

  void print() {
    for (auto &it : pads) {
      TC_P(it.first->node_type_name);
      TC_P(it.second.bounds[0]);
      TC_P(it.second.bounds[1]);
    }
  }
};

TLANG_NAMESPACE_END
