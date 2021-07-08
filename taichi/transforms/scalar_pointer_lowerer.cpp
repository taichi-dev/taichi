#include <algorithm>
#include <array>

#include "taichi/inc/constants.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/transforms/scalar_pointer_lowerer.h"

namespace taichi {
namespace lang {

ScalarPointerLowerer::ScalarPointerLowerer(SNode *leaf_snode,
                                           const std::vector<Stmt *> &indices,
                                           const SNodeOpType snode_op,
                                           const bool is_bit_vectorized,
                                           VecStatement *lowered)
    : indices_(indices),
      snode_op_(snode_op),
      is_bit_vectorized_(is_bit_vectorized),
      lowered_(lowered) {
  for (auto *s = leaf_snode; s != nullptr; s = s->parent) {
    snodes_.push_back(s);
  }
  // From root to leaf
  std::reverse(snodes_.begin(), snodes_.end());

  const int path_inc = (int)(snode_op_ != SNodeOpType::undefined);
  path_length_ = (int)snodes_.size() - 1 + path_inc;
}

void ScalarPointerLowerer::run() {
  // |start_bits| is the index of the starting bit for a coordinate
  // for a given SNode. It characterizes the relationship between a parent
  // and a child SNode: "parent.start = child.start + child.num_bits".
  //
  // For example, if there are two 1D snodes a and b,
  // where a = ti.root.dense(ti.i, 2) and b = a.dense(ti.i, 8),
  // we have a.start = b.start + 3 for the i-th dimension.
  // When accessing b[15], then bits [0, 3) of 15 are for accessing b,
  // and bit [3, 4) of 15 is for accessing a.
  std::array<int, taichi_max_num_indices> start_bits = {0};
  for (const auto *s : snodes_) {
    for (int j = 0; j < taichi_max_num_indices; j++) {
      start_bits[j] += s->extractors[j].num_bits;
    }
  }

  if (path_length_ == 0)
    return;

  Stmt *last = lowered_->push_back<GetRootStmt>(snodes_[0]);
  for (int i = 0; i < path_length_; i++) {
    auto *snode = snodes_[i];
    // TODO: Explain this condition
    if (is_bit_vectorized_ && (snode->type == SNodeType::bit_array) &&
        (i == path_length_ - 1) && (snodes_[i - 1]->type == SNodeType::dense)) {
      continue;
    }
    std::vector<Stmt *> lowered_indices;
    std::vector<int> strides;
    // extract bits
    for (int k_ = 0; k_ < (int)indices_.size(); k_++) {
      for (int k = 0; k < taichi_max_num_indices; k++) {
        if (snode->physical_index_position[k_] == k) {
          start_bits[k] -= snode->extractors[k].num_bits;
          const int begin = start_bits[k];
          const int end = begin + snode->extractors[k].num_bits;
          auto extracted = Stmt::make<BitExtractStmt>(indices_[k_], begin, end);
          lowered_indices.push_back(extracted.get());
          lowered_->push_back(std::move(extracted));
          strides.push_back(1 << snode->extractors[k].num_bits);
        }
      }
    }
    // linearize
    auto *linearized =
        lowered_->push_back<LinearizeStmt>(lowered_indices, strides);

    last = handle_snode_at_level(i, linearized, last);
  }
}

}  // namespace lang
}  // namespace taichi
