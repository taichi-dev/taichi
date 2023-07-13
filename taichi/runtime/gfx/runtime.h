#pragma once
#include "taichi/util/lang_util.h"

#include <vector>
#include <chrono>

#include "taichi/rhi/device.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/program/compile_config.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/kernel_launcher.h"

namespace taichi::lang {
namespace gfx {

using namespace taichi::lang::spirv;

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

using high_res_clock = std::chrono::high_resolution_clock;

// TODO: In the future this isn't necessarily a pointer, since DeviceAllocation
// is already a pretty cheap handle>
using InputBuffersMap =
    std::unordered_map<BufferInfo, DeviceAllocation *, BufferInfoHasher>;

class SNodeTreeManager;

class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<std::vector<uint32_t>> spirv_bins;
    std::size_t num_snode_trees{0};

    Device *device{nullptr};
    std::vector<DeviceAllocation *> root_buffers;
    DeviceAllocation *global_tmps_buffer{nullptr};
    DeviceAllocation *listgen_buffer{nullptr};

    PipelineCache *backend_cache{nullptr};
  };

  explicit CompiledTaichiKernel(const Params &ti_params);

  const TaichiKernelAttributes &ti_kernel_attribs() const;

  size_t num_pipelines() const;

  size_t get_args_buffer_size() const;
  size_t get_ret_buffer_size() const;

  Pipeline *get_pipeline(int i);

  DeviceAllocation *get_buffer_bind(const BufferInfo &bind) {
    return input_buffers_[bind];
  }

 private:
  TaichiKernelAttributes ti_kernel_attribs_;
  std::vector<TaskAttributes> tasks_attribs_;

  [[maybe_unused]] Device *device_;

  InputBuffersMap input_buffers_;

  size_t args_buffer_size_{0};
  size_t ret_buffer_size_{0};
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
};

class TI_DLL_EXPORT GfxRuntime {
 public:
  struct Params {
    Device *device{nullptr};
    KernelProfilerBase *profiler{nullptr};
  };

  explicit GfxRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~GfxRuntime();

  using KernelHandle = KernelLauncher::Handle;

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<std::vector<uint32_t>> task_spirv_source_codes;
    std::size_t num_snode_trees{0};
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  void launch_kernel(KernelHandle handle, LaunchContextBuilder &host_ctx);

  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size);
  void copy_image(DeviceAllocation dst,
                  DeviceAllocation src,
                  const ImageCopyParams &params);

  DeviceAllocation create_image(const ImageParams &params);
  void track_image(DeviceAllocation image, ImageLayout layout);
  void untrack_image(DeviceAllocation image);
  void transition_image(DeviceAllocation image, ImageLayout layout);

  void synchronize();

  StreamSemaphore flush();

  Device *get_ti_device() const;

  void add_root_buffer(size_t root_buffer_size);

  DeviceAllocation *get_root_buffer(int id) const;

  size_t get_root_buffer_size(int id) const;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs);

  bool used_in_kernel(DeviceAllocationId id) {
    return ndarrays_in_use_.count(id) > 0 || argpacks_in_use_.count(id) > 0;
  }

  static std::pair<const lang::StructType *, size_t>
  get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                   const std::string &layout);

  static std::tuple<const lang::StructType *, size_t, size_t>
  get_struct_type_with_data_layout_impl(const lang::StructType *old_ty,
                                        const std::string &layout);

 private:
  friend class taichi::lang::gfx::SNodeTreeManager;

  void ensure_current_cmdlist();
  void submit_current_cmdlist_if_timeout();

  void init_nonroot_buffers();

  Device *device_{nullptr};
  KernelProfilerBase *profiler_;

  std::unique_ptr<PipelineCache> backend_cache_{nullptr};

  std::vector<std::unique_ptr<DeviceAllocationGuard>> root_buffers_;
  std::unique_ptr<DeviceAllocationGuard> global_tmps_buffer_;
  // FIXME: Support proper multiple lists
  std::unique_ptr<DeviceAllocationGuard> listgen_buffer_;

  std::vector<std::unique_ptr<DeviceAllocationGuard>> ctx_buffers_;

  std::unique_ptr<CommandList> current_cmdlist_{nullptr};
  high_res_clock::time_point current_cmdlist_pending_since_;

  std::vector<std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;

  std::unordered_map<DeviceAllocation *, size_t> root_buffers_size_map_;
  std::unordered_map<DeviceAllocationId, ImageLayout> last_image_layouts_;
  // [Note] Why do we need to track ndarrays that are in use?
  // Since we separate cmdlist is async, taichi needs a way to know whether
  // ndarrays are still used by pending kernels to be executed. So we use
  // ndarray_in_use_ to track this so that we can free memory allocated for
  // ndarray whenever it's safe to do so.
  std::unordered_set<DeviceAllocationId> ndarrays_in_use_;
  std::unordered_set<DeviceAllocationId> argpacks_in_use_;
};

GfxRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    Arch arch,
    const DeviceCapabilityConfig &caps,
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    const CompileConfig &compile_config);

}  // namespace gfx
}  // namespace taichi::lang
