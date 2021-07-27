#include "taichi/struct/snode_tree.h"

#include "taichi/struct/struct.h"

namespace taichi {
namespace lang {

SNodeTree::SNodeTree(int id, std::unique_ptr<SNode> root, bool packed)
    : id_(id), root_(std::move(root)) {
  infer_snode_properties(*root_, packed);
}

}  // namespace lang
}  // namespace taichi
