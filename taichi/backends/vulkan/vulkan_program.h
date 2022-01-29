#pragma once
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "vk_mem_alloc.h"

#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/program.h"

#include <optional>

namespace taichi {
namespace lang {

namespace vulkan {
class VulkanDeviceCreator;
}

class VulkanProgramImpl : public ProgramImpl {
 public:
  VulkanProgramImpl(CompileConfig &config);
  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  // TODO: support sparse in vulkan
  }

  void compile_snode_tree_types(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees) override;

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree,
                              std::vector<std::unique_ptr<SNodeTree>> &,
                              uint64 *result_buffer) override;

  void synchronize() override {
    vulkan_runtime_->synchronize();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override;

  virtual void destroy_snode_tree(SNodeTree *snode_tree) override {
    vulkan_runtime_->destroy_snode_tree(snode_tree);
  }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;

  Device *get_compute_device() override {
    if (embedded_device_) {
      return embedded_device_->device();
    }
    return nullptr;
  }

  Device *get_graphics_device() override {
    if (embedded_device_) {
      return embedded_device_->device();
    }
    return nullptr;
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return vulkan_runtime_->get_snode_tree_device_ptr(tree_id);
  }

  ~VulkanProgramImpl();

 private:
  std::unique_ptr<vulkan::VulkanDeviceCreator> embedded_device_{nullptr};
  std::unique_ptr<vulkan::VkRuntime> vulkan_runtime_;
  std::vector<spirv::CompiledSNodeStructs> aot_compiled_snode_structs_;
};
}  // namespace lang
}  // namespace taichi
