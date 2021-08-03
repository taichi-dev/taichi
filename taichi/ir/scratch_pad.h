#pragma once

#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN

class Stmt;

enum AccessFlag : unsigned int {
  read = 1 << 1,
  write = 1 << 2,
  accumulate = 1 << 3
};

inline AccessFlag operator|(AccessFlag a, AccessFlag b) {
  return static_cast<AccessFlag>(static_cast<unsigned>(a) |
                                 static_cast<unsigned>(b));
}

inline AccessFlag operator&(AccessFlag a, AccessFlag b) {
  return static_cast<AccessFlag>(static_cast<unsigned>(a) &
                                 static_cast<unsigned>(b));
}

inline AccessFlag operator|=(AccessFlag &a, AccessFlag &b) {
  a = a | b;
  return a;
}

class ScratchPad {
 public:
  // The lowest and highest index in each dimension.
  struct BoundRange {
    int low{0};
    int high{0};

    int range() const {
      return high - low;
    }

    TI_IO_DEF(low, high);
  };

  SNode *snode{nullptr};
  using AccessFlag = taichi::lang::AccessFlag;

  std::vector<int> coefficients;
  std::vector<BoundRange> bounds;
  // pad_size[i] := bounds[i].high - bounds[i].low
  // TODO: This can be replaced by a function call to bounds[i].range()
  std::vector<int> pad_size;
  // block_size[i] := (1 << snode.extractor[i].num_bits)
  std::vector<int> block_size;
  bool finalized;
  int dim;
  bool empty;

  AccessFlag total_flags;
  std::vector<AccessFlag> flags;
  std::vector<std::pair<std::vector<int>, AccessFlag>> accesses;

  ScratchPad() = default;

  ScratchPad(SNode *snode) : snode(snode) {
    TI_ASSERT(snode != nullptr);
    dim = snode->num_active_indices;
    coefficients.resize(dim);
    bounds.resize(dim);
    pad_size.resize(dim);

    finalized = false;

    total_flags = AccessFlag(0);
    BoundRange init_bound;
    init_bound.low = std::numeric_limits<int>::max();
    init_bound.high = std::numeric_limits<int>::min();
    std::fill(bounds.begin(), bounds.end(), init_bound);
    empty = false;
  }

  void access(const std::vector<int> &coeffs,
              const std::vector<int> &indices,
              AccessFlag flags) {
    TI_ASSERT(!finalized);
    empty = true;
    TI_ASSERT((int)indices.size() == dim);
    for (int i = 0; i < dim; i++) {
      coefficients[i] = coeffs[i];
      bounds[i].low = std::min(bounds[i].low, indices[i]);
      bounds[i].high = std::max(bounds[i].high, indices[i] + 1);
      pad_size[i] = bounds[i].range();
    }
    accesses.emplace_back(indices, flags);
  }

  void finalize() {
    int size = 1;
    for (int i = 0; i < dim; i++) {
      size *= pad_size[i];
    }
    flags.resize(size);

    block_size.resize(dim);
    for (int i = 0; i < dim; i++) {
      block_size[i] =
          snode->parent->extractors[snode->physical_index_position[i]].shape;
      TI_ASSERT(bounds[i].low != std::numeric_limits<int>::max());
      TI_ASSERT(bounds[i].high != std::numeric_limits<int>::min());
    }

    finalized = true;
    flags = std::vector<AccessFlag>(pad_size_linear(), AccessFlag(0));

    for (auto &acc : accesses) {
      total_flags |= acc.second;
      flags[linearized_index(acc.first)] |= acc.second;
    }
  }

  void codegen_cpu() {
  }

  std::string name() {
    return snode->node_type_name + "_scratch_pad";
  }

  bool is_pure() const {
    return bit::is_power_of_two((unsigned)total_flags);
  }

  int pad_size_linear() {
    TI_ASSERT(finalized);
    int s = 1;
    for (int i = 0; i < dim; i++) {
      s *= pad_size[i];
    }
    return s;
  }

  int block_size_linear() {
    TI_ASSERT(finalized);
    int s = 1;
    for (int i = 0; i < dim; i++) {
      s *= block_size[i];
    }
    return s;
  }

  int linearized_index(const std::vector<int> &indices) {
    int ret = 0;
    TI_ASSERT(finalized);
    for (int i = 0; i < dim; i++) {
      ret *= (bounds[i].high - bounds[i].low);
      ret += indices[i] - bounds[i].low;
    }
    return ret;
  }

  std::string extract_offset(std::string var, int d) const {
    auto div = 1;
    for (int i = d + 1; i < dim; i++) {
      div *= pad_size[i];
    }
    return fmt::format("({} / {} % {} + {})", var, div, pad_size[d],
                       bounds[d].low);
  }

  /*
  std::string array_dimensions_str() const {
    std::string ret = "";
    for (int i = 0; i < dim; i++) {
      ret += fmt::format("[{}]", bounds[1][i] - bounds[0][i]);
    }
    return ret;
  }
   */

  std::string global_to_linearized_local(const std::vector<Stmt *> &loop_vars,
                                         const std::vector<Stmt *> &indices);
};

inline int div_floor(int a, int b) {
  return a >= 0 ? a / b : (a - b + 1) / b;
}

class ScratchPads {
 public:
  std::map<SNode *, ScratchPad> pads;

  using AccessFlag = ScratchPad::AccessFlag;

  void insert(SNode *snode) {
    if (pads.find(snode) == pads.end()) {
      pads.emplace(std::piecewise_construct, std::forward_as_tuple(snode),
                   std::forward_as_tuple(snode));
    } else {
      TI_ERROR("ScratchPad for {} already exists.", snode->node_type_name);
    }
  }

  void access(SNode *snode,
              const std::vector<int> &coeffs,
              const std::vector<int> &indices,
              AccessFlag flags) {
    TI_ASSERT(snode != nullptr);
    if (pads.find(snode) == pads.end())
      return;
    pads.find(snode)->second.access(coeffs, indices, flags);
    /*
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
    */
  }

  void finalize() {
    for (auto &pad : pads) {
      pad.second.finalize();
    }
  }

  void generate_address_code(SNode *snode, const std::vector<int> &indices) {
    if (pads.find(snode) != pads.end()) {
      auto &pad = pads[snode];
      int offset = 0;
      for (int i = 0; i < pad.dim; i++) {
        offset = offset + (indices[i] - pad.bounds[i].low);
        if (i > 0)
          offset = offset * pad.pad_size[i - 1];
      }
    } else if (pads.find(snode->parent) != pads.end()) {
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void print() {
    for (auto &it : pads) {
      TI_P(it.first->node_type_name);
      TI_P(it.second.bounds);
    }
  }

  bool has(SNode *snode) {
    return pads.find(snode) != pads.end();
  }

  ScratchPad &get(SNode *snode) {
    TI_ASSERT(pads.find(snode) != pads.end());
    return pads[snode];
  }
};

TLANG_NAMESPACE_END
