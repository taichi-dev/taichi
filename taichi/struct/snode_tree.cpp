#include "taichi/struct/snode_tree.h"
#include "taichi/struct/struct.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

SNodeTree::SNodeTree(int id, std::unique_ptr<SNode> root, Program *prog)
    : id_(id), root_(std::move(root)), prog(prog) {
  infer_snode_properties(*root_);
}

void SNodeTree::destroy() {
  if (destroyed) {
    TI_ERROR("SNode tree {} has been destroyed", id_);
  }
  prog->snode_tree_buffer_manager->destroy(id_);
  destroyed = true;
}

}  // namespace lang
}  // namespace taichi
