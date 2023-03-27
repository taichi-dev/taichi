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
#include "vk_mem_alloc.h"

#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/program.h"

#include <optional>

namespace taichi::lang {

namespace vulkan {
class VulkanDeviceCreator;
}

class VulkanProgramImpl : public ProgramImpl {
 public:
  explicit VulkanProgramImpl(CompileConfig &config);
  FunctionType compile(const CompileConfig &compile_config,
                       Kernel *kernel) override;

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
    if (vulkan_runtime_) {
      vulkan_runtime_->synchronize();
    }
  }

  StreamSemaphore flush() override {
    return vulkan_runtime_->flush();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) override;

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_ASSERT(snode_tree_mgr_ != nullptr);
    snode_tree_mgr_->destroy_snode_tree(snode_tree);
  }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;
  bool used_in_kernel(DeviceAllocationId id) override {
    return vulkan_runtime_->used_in_kernel(id);
  }
  DeviceAllocation allocate_texture(const ImageParams &params) override;

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

  size_t get_field_in_tree_offset(int tree_id, const SNode *child) override {
    return snode_tree_mgr_->get_field_in_tree_offset(tree_id, child);
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return snode_tree_mgr_->get_snode_tree_device_ptr(tree_id);
  }

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;

  std::pair<const StructType *, size_t> get_struct_type_with_data_layout(
      const StructType *old_ty,
      const std::string &layout) override {
    // Ported from KernelContextAttributes::KernelContextAttributes as is.
    // TODO: refactor this.
    TI_TRACE("get_struct_type_with_data_layout: {}", layout);
    auto is_ret = layout[0] == 'r';
    auto has_buffer_ptr = layout[1] == 'b';
    auto members = old_ty->elements();
    size_t bytes = 0;
    for (int i = 0; i < members.size(); i++) {
      auto &member = members[i];
      const Type *element_type;
      size_t stride = 0;
      bool is_array = false;
      if (!is_ret) {
        element_type = DataType(member.type).ptr_removed().get_element_type();
        is_array = member.type->is<PointerType>();
      } else if (auto tensor_type = member.type->cast<TensorType>()) {
        element_type = tensor_type->get_element_type();
        stride = tensor_type->get_num_elements() * data_type_size(element_type);
        is_array = true;
      } else {
        auto primitive_type = member.type->as<PrimitiveType>();
        element_type = primitive_type;
        stride = data_type_size(primitive_type);
      }

      const size_t dt_bytes = (member.type->is<PointerType>() && has_buffer_ptr)
                                  ? sizeof(uint64_t)
                                  : data_type_size(element_type);
      TI_TRACE("dt_bytes={} stride={} is_array={} is_ret={}", dt_bytes, stride,
               is_array, is_ret);
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      member.offset = bytes;
      bytes += is_ret ? stride : dt_bytes;
      TI_TRACE("  at={} {} offset_in_mem={} stride={}",
               is_array ? (is_ret ? "array" : "vector ptr") : "scalar", i,
               member.offset, stride);
    }
    if (!is_ret) {
      bytes = (bytes + 4 - 1) / 4 * 4;
    }
    TI_TRACE("  total_bytes={}", bytes);
    return {TypeFactory::get_instance()
                .get_struct_type(members, layout)
                ->as<StructType>(),
            bytes};
  }

  std::string get_kernel_return_data_layout() override {
    return "r-";
  };

  std::string get_kernel_argument_data_layout() override {
    auto has_buffer_ptr = vulkan_runtime_->get_ti_device()->get_caps().get(
        DeviceCapability::spirv_has_physical_storage_buffer);
    return "a" + std::string(has_buffer_ptr ? "b" : "-");
  };

  ~VulkanProgramImpl() override;

 protected:
  std::unique_ptr<KernelCompiler> make_kernel_compiler() override;

 private:
  std::unique_ptr<vulkan::VulkanDeviceCreator> embedded_device_{nullptr};
  std::unique_ptr<gfx::GfxRuntime> vulkan_runtime_{nullptr};
  std::unique_ptr<gfx::SNodeTreeManager> snode_tree_mgr_{nullptr};
  std::vector<spirv::CompiledSNodeStructs> aot_compiled_snode_structs_;
};
}  // namespace taichi::lang
