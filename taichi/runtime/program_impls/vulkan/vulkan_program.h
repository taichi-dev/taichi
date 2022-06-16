#pragma once
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
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
    return 0;  // TODO: support sparse
  }

  void compile_snode_tree_types(SNodeTree *tree) override;

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  void synchronize() override {
    vulkan_runtime_->synchronize();
  }

  StreamSemaphore flush() override {
    return vulkan_runtime_->flush();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override;

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_ASSERT(snode_tree_mgr_ != nullptr);
    snode_tree_mgr_->destroy_snode_tree(snode_tree);
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
    return snode_tree_mgr_->get_snode_tree_device_ptr(tree_id);
  }

  std::unique_ptr<aot::Kernel> make_aot_kernel(Kernel &kernel) override;

  ~VulkanProgramImpl();

 private:
  std::unique_ptr<vulkan::VulkanDeviceCreator> embedded_device_{nullptr};
  std::unique_ptr<gfx::GfxRuntime> vulkan_runtime_{nullptr};
  std::unique_ptr<gfx::SNodeTreeManager> snode_tree_mgr_{nullptr};
  std::vector<spirv::CompiledSNodeStructs> aot_compiled_snode_structs_;
};
}  // namespace lang
}  // namespace taichi
