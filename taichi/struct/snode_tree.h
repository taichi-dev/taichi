#pragma once

#include <memory>

#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {

class Program;

/**
 * Represents a tree of SNodes.
 *
 * An SNodeTree will be backed by a contiguous chunk of memory.
 */
class SNodeTree {
 public:
  constexpr static int kFirstID = 0;
  Program *prog;

  /**
   * Constructor.
   *
   * @param id Id of the tree
   * @param root Root of the tree
   */
  explicit SNodeTree(int id,
                     std::unique_ptr<SNode> root,
                     bool packed,
                     Program *prog);

  void destroy();

  int id() const {
    if (destroyed) {
      TI_ERROR("SNode tree {} has been destroyed", id_);
    }
    return id_;
  }

  const SNode *root() const {
    if (destroyed) {
      TI_ERROR("SNode tree {} has been destroyed", id_);
    }
    return root_.get();
  }

  SNode *root() {
    if (destroyed) {
      TI_ERROR("SNode tree {} has been destroyed", id_);
    }
    return root_.get();
  }

 private:
  int id_{0};
  std::unique_ptr<SNode> root_{nullptr};
  bool destroyed{false};
};

}  // namespace lang
}  // namespace taichi
