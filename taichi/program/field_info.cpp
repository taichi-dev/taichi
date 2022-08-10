#include "taichi/program/field_info.h"

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

  return root_ptr.get_ptr(program->get_field_in_tree_offset(tree_id, snode));
}

}  // namespace ui

}  // namespace taichi
