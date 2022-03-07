#pragma once

#include <vector>

#include "taichi/backends/device.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/struct/snode_tree.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VkRuntime;

/**
 * @brief Manages the SNodeTrees for the Vulkan backend.
 *
 */
class SNodeTreeManager {
 private:
  using CompiledSNodeStructs = taichi::lang::spirv::CompiledSNodeStructs;

 public:
  explicit SNodeTreeManager(VkRuntime *rtm);

  const std::vector<CompiledSNodeStructs> &get_compiled_structs() const {
    return compiled_snode_structs_;
  }

  void materialize_snode_tree(SNodeTree *tree);

  void destroy_snode_tree(SNodeTree *snode_tree);

  DevicePtr get_snode_tree_device_ptr(int tree_id);

 private:
  VkRuntime *const runtime_;
  std::vector<CompiledSNodeStructs> compiled_snode_structs_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
