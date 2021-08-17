#include "taichi/struct/snode_tree.h"

namespace taichi {
namespace lang {

SNodeTree::SNodeTree(int id, std::unique_ptr<SNode> root)
    : id_(id), root_(std::move(root)) {
  check_tree_validity(*root_);
}

void SNodeTree::check_tree_validity(SNode &node) {
  if (node.ch.empty()) {
    if (node.type != SNodeType::place && node.type != SNodeType::root) {
      TI_ERROR("{} node must have at least one child.",
               snode_type_name(node.type));
    }
  }
  for (auto &ch : node.ch) {
    check_tree_validity(*ch);
  }
}

}  // namespace lang
}  // namespace taichi
