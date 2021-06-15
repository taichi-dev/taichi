#pragma once

#include <memory>

#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {

/**
 * Represents a tree of SNodes.
 *
 * An SNodeTree will be backed by a contiguous chunk of memory.
 */
class SNodeTree {
 public:
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
};

}  // namespace lang
}  // namespace taichi
