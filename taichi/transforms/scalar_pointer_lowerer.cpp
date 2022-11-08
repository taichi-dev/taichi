#include <algorithm>
#include <array>

#include "taichi/inc/constants.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/transforms/scalar_pointer_lowerer.h"
#include "taichi/transforms/utils.h"

namespace taichi::lang {

ScalarPointerLowerer::ScalarPointerLowerer(SNode *leaf_snode,
                                           const std::vector<Stmt *> &indices,
                                           const SNodeOpType snode_op,
                                           const bool is_bit_vectorized,
                                           VecStatement *lowered,
                                           const bool packed)
    : indices_(indices),
      snode_op_(snode_op),
      is_bit_vectorized_(is_bit_vectorized),
      lowered_(lowered),
      packed_(packed) {
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
  // general shape calculation - no dependence on POT
  std::array<int, taichi_max_num_indices> total_shape;
  total_shape.fill(1);
  for (const auto *s : snodes_) {
    for (int j = 0; j < taichi_max_num_indices; j++) {
      total_shape[j] *= s->extractors[j].shape;
    }
  }
  std::array<bool, taichi_max_num_indices> is_first_extraction;
  is_first_extraction.fill(true);

  if (path_length_ == 0)
    return;

  auto *leaf_snode = snodes_[path_length_ - 1];
  Stmt *last = lowered_->push_back<GetRootStmt>(snodes_[0]);
  for (int i = 0; i < path_length_; i++) {
    auto *snode = snodes_[i];
    // TODO: Explain this condition
    if (is_bit_vectorized_ && (snode->type == SNodeType::quant_array) &&
        (i == path_length_ - 1) && (snodes_[i - 1]->type == SNodeType::dense)) {
      continue;
    }
    std::vector<Stmt *> lowered_indices;
    std::vector<int> strides;
    // extract lowered indices
    for (int k_ = 0; k_ < (int)indices_.size(); k_++) {
      int k = leaf_snode->physical_index_position[k_];
      if (!snode->extractors[k].active)
        continue;
      Stmt *extracted;
      if (packed_) {  // no dependence on POT
        const int prev = total_shape[k];
        total_shape[k] /= snode->extractors[k].shape;
        const int next = total_shape[k];
        if (is_first_extraction[k]) {
          // Upon first extraction on axis k, "indices_[k_]" is the user
          // coordinate on axis k and "prev" is the total shape of axis k.
          // Unless it is an invalid out-of-bound access, we can assume
          // "indices_[k_] < prev" so we don't need a mod here.
          auto const_next = lowered_->push_back<ConstStmt>(TypedConstant(next));
          extracted = lowered_->push_back<BinaryOpStmt>(
              BinaryOpType::div, indices_[k_], const_next);
        } else {
          extracted = generate_mod_x_div_y(lowered_, indices_[k_], prev, next);
        }
      } else {
        const int end = start_bits[k];
        start_bits[k] -= snode->extractors[k].num_bits;
        const int begin = start_bits[k];
        if (is_first_extraction[k] && begin == 0) {
          // Similar optimization as above. In this case the full user
          // coordinate is extracted so we don't need a BitExtractStmt.
          extracted = indices_[k_];
        } else {
          extracted =
              lowered_->push_back<BitExtractStmt>(indices_[k_], begin, end);
        }
      }
      is_first_extraction[k] = false;
      lowered_indices.push_back(extracted);
      strides.push_back(snode->extractors[k].shape);
    }
    // linearize
    auto *linearized =
        lowered_->push_back<LinearizeStmt>(lowered_indices, strides);

    last = handle_snode_at_level(i, linearized, last);
  }
}

}  // namespace taichi::lang
