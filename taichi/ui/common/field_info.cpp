#include "taichi/ui/common/field_info.h"
#include "taichi/program/program.h"

namespace taichi {

namespace ui {

using namespace taichi::lang;

DevicePtr get_device_ptr(taichi::lang::Program *program, SNode *snode) {
  /*
  GGUI makes the assumption that the input fields are created directly from
  ti.field() or ti.Vector field with `shape` specified. In other words, we
  assume that the fields are created via ti.root.dense.place() That is, the
  parent of the snode is a dense, and the parent of that node is a root. Note
  that, GGUI's python-side code creates a staging buffer to construct the VBO,
  which obeys this assumption. Thus, the only situation where this assumption
  may be violated is for set_image(), because the image isn't part of the VBO.
  Using this assumption, we will compute the offset of this field relative to
  the begin of the root buffer.
  */

  SNode *dense_parent = snode->parent;
  SNode *root = dense_parent->parent;

  int tree_id = root->get_snode_tree_id();
  DevicePtr root_ptr = program->get_snode_tree_device_ptr(tree_id);

  int64 offset = 0;

  int child_id = root->child_id(dense_parent);

  TI_ASSERT_INFO(root == program->get_snode_root(tree_id),
                 "SNode roots don't match");

  for (int i = 0; i < child_id; ++i) {
    SNode *child = root->ch[i].get();
    offset += child->cell_size_bytes * child->num_cells_per_container;
  }

  return root_ptr.get_ptr(offset);
}

}  // namespace ui

}  // namespace taichi
