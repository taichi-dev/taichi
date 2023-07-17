#pragma once
#include "taichi/aot/module_loader.h"
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/program.h"
#include "taichi/runtime/program_impls/gfx/gfx_program.h"

#include <optional>

namespace taichi::lang {

namespace vulkan {
class VulkanDeviceCreator;
}

class VulkanProgramImpl : public GfxProgramImpl {
 public:
  explicit VulkanProgramImpl(CompileConfig &config);
  ~VulkanProgramImpl() override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

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

  void finalize() override;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;

 private:
  std::unique_ptr<vulkan::VulkanDeviceCreator> embedded_device_{nullptr};
};
}  // namespace taichi::lang
