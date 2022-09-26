#pragma once

#include <memory>
#include <unordered_map>

#include "taichi/ir/snode.h"

namespace taichi::lang {

/**
 * Represents a tree of SNodes.
 *
 * An SNodeTree will be backed by a contiguous chunk of memory.
 */
class SNodeTree {
 public:
  constexpr static int kFirstID = 0;

  /**
   * Constructor.
   *
   * @param id Id of the tree
   * @param root Root of the tree
   */
  explicit SNodeTree(int id, std::unique_ptr<SNode> root);

  int id() const {
    return id_;
  }

  const SNode *root() const {
    return root_.get();
  }

  SNode *root() {
    return root_.get();
  }

 private:
  int id_{0};
  std::unique_ptr<SNode> root_{nullptr};

  void check_tree_validity(SNode &node);
};

/**
 * Returns the mapping from each SNode under @param root to itself.
 *
 * @param root Root SNode
 * @returns The ID mapping
 */
std::unordered_map<int, int> get_snodes_to_root_id(const SNode &root);

}  // namespace taichi::lang
