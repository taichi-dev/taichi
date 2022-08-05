#include "taichi/runtime/gfx/snode_tree_manager.h"

#include "taichi/runtime/gfx/runtime.h"

namespace taichi {
namespace lang {
namespace gfx {

SNodeTreeManager::SNodeTreeManager(GfxRuntime *rtm) : runtime_(rtm) {
}

void SNodeTreeManager::materialize_snode_tree(SNodeTree *tree) {
  auto *const root = tree->root();
  CompiledSNodeStructs compiled_structs = compile_snode_structs(*root);
  runtime_->add_root_buffer(compiled_structs.root_size);
  compiled_snode_structs_.push_back(compiled_structs);
}

void SNodeTreeManager::destroy_snode_tree(SNodeTree *snode_tree) {
  int root_id = -1;
  for (int i = 0; i < compiled_snode_structs_.size(); ++i) {
    if (compiled_snode_structs_[i].root == snode_tree->root()) {
      root_id = i;
    }
  }
  if (root_id == -1) {
    TI_ERROR("the tree to be destroyed cannot be found");
  }
  runtime_->root_buffers_[root_id].reset();
}

size_t SNodeTreeManager::get_field_in_tree_offset(int tree_id,
                                                  const SNode *child) {
  auto &snode_struct = compiled_snode_structs_[tree_id];
  TI_ASSERT_INFO(
      snode_struct.snode_descriptors.find(child->id) !=
              snode_struct.snode_descriptors.end() &&
          snode_struct.snode_descriptors.at(child->id).snode == child,
      "Requested SNode not found in compiled SNodeTree");

  size_t offset = 0;
  for (const SNode *sn = child; sn; sn = sn->parent) {
    offset +=
        snode_struct.snode_descriptors.at(sn->id).mem_offset_in_parent_cell;
  }

  return offset;
}

DevicePtr SNodeTreeManager::get_snode_tree_device_ptr(int tree_id) {
  return runtime_->root_buffers_[tree_id]->get_ptr();
}

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
