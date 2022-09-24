#include "taichi/struct/snode_tree.h"

namespace taichi::lang {
namespace {

void get_snodes_to_root_id_impl(const SNode &node,
                                const int root_id,
                                std::unordered_map<int, int> *map) {
  (*map)[node.id] = root_id;
  for (auto &ch : node.ch) {
    get_snodes_to_root_id_impl(*ch, root_id, map);
  }
}

}  // namespace

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

std::unordered_map<int, int> get_snodes_to_root_id(const SNode &root) {
  // TODO: Consider generalizing this SNode visiting method
  std::unordered_map<int, int> res;
  get_snodes_to_root_id_impl(root, root.id, &res);
  return res;
}

}  // namespace taichi::lang
