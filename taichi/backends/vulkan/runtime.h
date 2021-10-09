#pragma once
#include "taichi/lang_util.h"

#include <vector>

#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"
#include "taichi/backends/vulkan/kernel_utils.h"
#include "taichi/program/compile_config.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"

namespace taichi {
namespace lang {
namespace vulkan {

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

// TODO: In the future this isn't necessarily a pointer, since DeviceAllocation
// is already a pretty cheap handle>
using InputBuffersMap =
    std::unordered_map<BufferInfo, DeviceAllocation *, BufferInfoHasher>;

class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<std::vector<uint32_t>> spirv_bins;
    std::vector<CompiledSNodeStructs> compiled_structs;

    Device *device{nullptr};
    std::vector<DeviceAllocation *> root_buffers;
    DeviceAllocation *global_tmps_buffer{nullptr};
  };

  CompiledTaichiKernel(const Params &ti_params);

  const TaichiKernelAttributes &ti_kernel_attribs() const;

  size_t num_pipelines() const;

  DeviceAllocation *ctx_buffer() const;

  DeviceAllocation *ctx_buffer_host() const;

  void command_list(CommandList *cmdlist) const;

 private:
  TaichiKernelAttributes ti_kernel_attribs_;
  std::vector<TaskAttributes> tasks_attribs_;

  Device *device_;

  InputBuffersMap input_buffers_;

  std::unique_ptr<DeviceAllocationGuard> ctx_buffer_{nullptr};
  std::unique_ptr<DeviceAllocationGuard> ctx_buffer_host_{nullptr};
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
};

class VkRuntime {
 public:
  struct Params {
    uint64_t *host_result_buffer{nullptr};
    Device *device{nullptr};
  };

  explicit VkRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~VkRuntime();

  class KernelHandle {
   private:
    friend class VkRuntime;
    int id_ = -1;
  };

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<std::vector<uint32_t>> task_spirv_source_codes;
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  void launch_kernel(KernelHandle handle, Context *host_ctx);

  void materialize_snode_tree(SNodeTree *tree);

  void destroy_snode_tree(SNodeTree *snode_tree);

  void synchronize();

  Device *get_ti_device() const;

  const std::vector<CompiledSNodeStructs> &get_compiled_structs() const;

 private:
  void init_buffers();
  void add_root_buffer(size_t root_buffer_size);

  Device *device_;

  uint64_t *const host_result_buffer_;

  std::vector<std::unique_ptr<DeviceAllocationGuard>> root_buffers_;
  std::unique_ptr<DeviceAllocationGuard> global_tmps_buffer_;

  std::unique_ptr<CommandList> current_cmdlist_{nullptr};

  std::vector<std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;

  std::vector<CompiledSNodeStructs> compiled_snode_structs_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
