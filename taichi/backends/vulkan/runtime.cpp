#include "taichi/backends/vulkan/runtime.h"

#include <chrono>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fp16.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace vulkan {

namespace {
class StopWatch {
 public:
  StopWatch() : begin_(std::chrono::system_clock::now()) {
  }

  int GetMicros() {
    typedef std::chrono::duration<float> fsec;

    auto now = std::chrono::system_clock::now();

    fsec fs = now - begin_;
    begin_ = now;
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(fs);
    return d.count();
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> begin_;
};

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           Context *host_ctx,
                           Device *device,
                           uint64_t *host_result_buffer,
                           DeviceAllocation *device_buffer,
                           DeviceAllocation *host_shadow_buffer)
      : ctx_attribs_(ctx_attribs),
        host_ctx_(host_ctx),
        host_result_buffer_(host_result_buffer),
        device_buffer_(device_buffer),
        host_shadow_buffer_(host_shadow_buffer),
        device_(device) {
  }

  void host_to_device() {
    if (ctx_attribs_->empty()) {
      return;
    }

    char *const device_base =
        reinterpret_cast<char *>(device_->map(*device_buffer_));

#define TO_DEVICE(short_type, type)                    \
  if (dt->is_primitive(PrimitiveTypeID::short_type)) { \
    auto d = host_ctx_->get_arg<type>(i);              \
    reinterpret_cast<type *>(device_ptr)[0] = d;       \
    break;                                             \
  }

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      const auto dt = arg.dt;
      char *device_ptr = device_base + arg.offset_in_mem;
      do {
        if (arg.is_array) {
          const void *host_ptr = host_ctx_->get_arg<void *>(i);
          std::memcpy(device_ptr, host_ptr, arg.stride);
          break;
        }
        if (device_->get_cap(DeviceCapability::spirv_has_int8)) {
          TO_DEVICE(i8, int8)
          TO_DEVICE(u8, uint8)
        }
        if (device_->get_cap(DeviceCapability::spirv_has_int16)) {
          TO_DEVICE(i16, int16)
          TO_DEVICE(u16, uint16)
        }
        TO_DEVICE(i32, int32)
        TO_DEVICE(u32, uint32)
        TO_DEVICE(f32, float32)
        if (device_->get_cap(DeviceCapability::spirv_has_int64)) {
          TO_DEVICE(i64, int64)
          TO_DEVICE(u64, uint64)
        }
        if (device_->get_cap(DeviceCapability::spirv_has_float64)) {
          TO_DEVICE(f64, float64)
        }
        if (device_->get_cap(DeviceCapability::spirv_has_float16)) {
          if (dt->is_primitive(PrimitiveTypeID::f16)) {
            auto d = fp16_ieee_from_fp32_value(host_ctx_->get_arg<float>(i));
            reinterpret_cast<uint16 *>(device_ptr)[0] = d;
            break;
          }
        }
        TI_ERROR("Vulkan does not support arg type={}", data_type_name(arg.dt));
      } while (0);
    }
    char *device_ptr = device_base + ctx_attribs_->extra_args_mem_offset();
    std::memcpy(device_ptr, host_ctx_->extra_args,
                ctx_attribs_->extra_args_bytes());

    device_->unmap(*device_buffer_);
#undef TO_DEVICE
  }

  void device_to_host() {
    if (ctx_attribs_->empty()) {
      return;
    }

    bool require_sync = ctx_attribs_->rets().size() > 0;
    if (!require_sync) {
      for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
        const auto &arg = ctx_attribs_->args()[i];
        if (arg.is_array) {
          require_sync = true;
        }
      }
    }

    if (require_sync) {
      device_->get_compute_stream()->command_sync();
    } else {
      return;
    }

    char *const device_base =
        reinterpret_cast<char *>(device_->map(*host_shadow_buffer_));

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      char *device_ptr = device_base + arg.offset_in_mem;
      if (arg.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);
      }
    }

#define TO_HOST(short_type, type)                          \
  if (dt->is_primitive(PrimitiveTypeID::short_type)) {     \
    const type d = *reinterpret_cast<type *>(device_ptr);  \
    host_result_buffer_[i] =                               \
        taichi_union_cast_with_different_sizes<uint64>(d); \
    break;                                                 \
  }

    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = device_base + ret.offset_in_mem;
      const auto dt = ret.dt;
      do {
        if (ret.is_array) {
          void *host_ptr = host_ctx_->get_arg<void *>(i);
          std::memcpy(host_ptr, device_ptr, ret.stride);
          break;
        }
        if (device_->get_cap(DeviceCapability::spirv_has_int8)) {
          TO_HOST(i8, int8)
          TO_HOST(u8, uint8)
        }
        if (device_->get_cap(DeviceCapability::spirv_has_int16)) {
          TO_HOST(i16, int16)
          TO_HOST(u16, uint16)
        }
        TO_HOST(i32, int32)
        TO_HOST(u32, uint32)
        TO_HOST(f32, float32)
        if (device_->get_cap(DeviceCapability::spirv_has_int64)) {
          TO_HOST(i64, int64)
          TO_HOST(u64, uint64)
        }
        if (device_->get_cap(DeviceCapability::spirv_has_float64)) {
          TO_HOST(f64, float64)
        }
        if (device_->get_cap(DeviceCapability::spirv_has_float16)) {
          if (dt->is_primitive(PrimitiveTypeID::f16)) {
            const float d = fp16_ieee_to_fp32_value(
                *reinterpret_cast<uint16 *>(device_ptr));
            host_result_buffer_[i] =
                taichi_union_cast_with_different_sizes<uint64>(d);
            break;
          }
        }
        TI_ERROR("Vulkan does not support return value type={}",
                 data_type_name(ret.dt));
      } while (0);
    }
#undef TO_HOST

    device_->unmap(*host_shadow_buffer_);
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(
      const KernelContextAttributes *ctx_attribs,
      Context *host_ctx,
      Device *device,
      uint64_t *host_result_buffer,
      DeviceAllocation *device_buffer,
      DeviceAllocation *host_shadow_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, host_ctx, device, host_result_buffer, device_buffer,
        host_shadow_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  Context *const host_ctx_;
  uint64_t *const host_result_buffer_;
  DeviceAllocation *const device_buffer_;
  DeviceAllocation *const host_shadow_buffer_;
  Device *const device_;
};

}  // namespace

// Info for launching a compiled Taichi kernel, which consists of a series of
// Vulkan pipelines.

CompiledTaichiKernel::CompiledTaichiKernel(const Params &ti_params)
    : ti_kernel_attribs_(*ti_params.ti_kernel_attribs),
      device_(ti_params.device) {
  input_buffers_[BufferType::GlobalTmps] = ti_params.global_tmps_buffer;
  for (int root = 0; root < ti_params.compiled_structs.size(); ++root) {
    BufferInfo buffer = {BufferType::Root, root};
    input_buffers_[buffer] = ti_params.root_buffers[root];
  }
  const auto ctx_sz = ti_kernel_attribs_.ctx_attribs.total_bytes();
  if (!ti_kernel_attribs_.ctx_attribs.empty()) {
    ctx_buffer_ = ti_params.device->allocate_memory_unique(
        {size_t(ctx_sz),
         /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage});
    ctx_buffer_host_ = ti_params.device->allocate_memory_unique(
        {size_t(ctx_sz),
         /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::Storage});
    input_buffers_[BufferType::Context] = ctx_buffer_.get();
  }

  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
  const auto &spirv_bins = ti_params.spirv_bins;
  TI_ASSERT(task_attribs.size() == spirv_bins.size());

  for (int i = 0; i < task_attribs.size(); ++i) {
    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary,
                                   (void *)spirv_bins[i].data(),
                                   spirv_bins[i].size() * sizeof(uint32_t)};
    auto vp =
        ti_params.device->create_pipeline(source_desc, ti_kernel_attribs_.name);
    pipelines_.push_back(std::move(vp));
  }
}

const TaichiKernelAttributes &CompiledTaichiKernel::ti_kernel_attribs() const {
  return ti_kernel_attribs_;
}

size_t CompiledTaichiKernel::num_pipelines() const {
  return pipelines_.size();
}

DeviceAllocation *CompiledTaichiKernel::ctx_buffer() const {
  return ctx_buffer_.get();
}

DeviceAllocation *CompiledTaichiKernel::ctx_buffer_host() const {
  return ctx_buffer_host_.get();
}

void CompiledTaichiKernel::command_list(CommandList *cmdlist) const {
  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;

  for (int i = 0; i < task_attribs.size(); ++i) {
    const auto &attribs = task_attribs[i];
    auto vp = pipelines_[i].get();
    const int group_x = (attribs.advisory_total_num_threads +
                         attribs.advisory_num_threads_per_group - 1) /
                        attribs.advisory_num_threads_per_group;
    ResourceBinder *binder = vp->resource_binder();
    for (auto &bind : attribs.buffer_binds) {
      binder->rw_buffer(0, bind.binding, *input_buffers_.at(bind.buffer));
    }

    cmdlist->bind_pipeline(vp);
    cmdlist->bind_resources(binder);
    cmdlist->dispatch(group_x);
    cmdlist->memory_barrier();
  }

  const auto ctx_sz = ti_kernel_attribs_.ctx_attribs.total_bytes();
  if (!ti_kernel_attribs_.ctx_attribs.empty()) {
    cmdlist->buffer_copy(ctx_buffer_host_->get_ptr(0), ctx_buffer_->get_ptr(0),
                         ctx_sz);
    cmdlist->buffer_barrier(*ctx_buffer_host_);
  }
}

VkRuntime::VkRuntime(const Params &params)
    : device_(params.device), host_result_buffer_(params.host_result_buffer) {
  TI_ASSERT(host_result_buffer_ != nullptr);
  init_buffers();
}

VkRuntime::~VkRuntime() {
  {
    decltype(ti_kernels_) tmp;
    tmp.swap(ti_kernels_);
  }
  global_tmps_buffer_.reset();
}

void VkRuntime::materialize_snode_tree(SNodeTree *tree) {
  auto *const root = tree->root();
  CompiledSNodeStructs compiled_structs = vulkan::compile_snode_structs(*root);
  add_root_buffer(compiled_structs.root_size);
  compiled_snode_structs_.push_back(compiled_structs);
}

void VkRuntime::destroy_snode_tree(SNodeTree *snode_tree) {
  int root_id = -1;
  for (int i = 0; i < compiled_snode_structs_.size(); ++i) {
    if (compiled_snode_structs_[i].root == snode_tree->root()) {
      root_id = i;
    }
  }
  if (root_id == -1) {
    TI_ERROR("the tree to be destroyed cannot be found");
  }
  root_buffers_[root_id].reset();
}

const std::vector<CompiledSNodeStructs> &VkRuntime::get_compiled_structs()
    const {
  return compiled_snode_structs_;
}

VkRuntime::KernelHandle VkRuntime::register_taichi_kernel(
    VkRuntime::RegisterParams reg_params) {
  CompiledTaichiKernel::Params params;
  params.ti_kernel_attribs = &(reg_params.kernel_attribs);
  params.compiled_structs = get_compiled_structs();
  params.device = device_;
  params.root_buffers = {};
  for (int root = 0; root < root_buffers_.size(); ++root) {
    params.root_buffers.push_back(root_buffers_[root].get());
  }
  params.global_tmps_buffer = global_tmps_buffer_.get();

  for (int i = 0; i < reg_params.task_spirv_source_codes.size(); ++i) {
    const auto &spirv_src = reg_params.task_spirv_source_codes[i];

    // If we can reach here, we have succeeded. Otherwise
    // std::optional::value() would have killed us.
    params.spirv_bins.push_back(std::move(spirv_src));
  }
  KernelHandle res;
  res.id_ = ti_kernels_.size();
  ti_kernels_.push_back(std::make_unique<CompiledTaichiKernel>(params));
  return res;
}

void VkRuntime::launch_kernel(KernelHandle handle, Context *host_ctx) {
  auto *ti_kernel = ti_kernels_[handle.id_].get();
  auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
      &ti_kernel->ti_kernel_attribs().ctx_attribs, host_ctx, device_,
      host_result_buffer_, ti_kernel->ctx_buffer(),
      ti_kernel->ctx_buffer_host());
  if (ctx_blitter) {
    TI_ASSERT(ti_kernel->ctx_buffer() != nullptr);
    ctx_blitter->host_to_device();
  }

  if (!current_cmdlist_) {
    current_cmdlist_ = device_->get_compute_stream()->new_command_list();
  }

  ti_kernel->command_list(current_cmdlist_.get());

  if (ctx_blitter) {
    device_->get_compute_stream()->submit(current_cmdlist_.get());
    ctx_blitter->device_to_host();

    current_cmdlist_ = nullptr;
  }
}

void VkRuntime::synchronize() {
  if (current_cmdlist_) {
    device_->get_compute_stream()->submit(current_cmdlist_.get());
    current_cmdlist_ = nullptr;
  }
  device_->get_compute_stream()->command_sync();
}

Device *VkRuntime::get_ti_device() const {
  return device_;
}

void VkRuntime::init_buffers() {
  size_t gtmp_buffer_size = 1024 * 1024;

  global_tmps_buffer_ = device_->allocate_memory_unique(
      {gtmp_buffer_size,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});

  // Need to zero fill the buffers, otherwise there could be NaN.
  Stream *stream = device_->get_compute_stream();
  auto cmdlist = stream->new_command_list();

  cmdlist->buffer_fill(global_tmps_buffer_->get_ptr(0), gtmp_buffer_size,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());
}

void VkRuntime::add_root_buffer(size_t root_buffer_size) {
  if (root_buffer_size == 0) {
    root_buffer_size = 4;  // there might be empty roots
  }
  std::unique_ptr<DeviceAllocationGuard> new_buffer =
      device_->allocate_memory_unique(
          {root_buffer_size,
           /*host_write=*/false, /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::Storage});
  Stream *stream = device_->get_compute_stream();
  auto cmdlist = stream->new_command_list();
  cmdlist->buffer_fill(new_buffer->get_ptr(0), root_buffer_size, /*data=*/0);
  stream->submit_synced(cmdlist.get());
  root_buffers_.push_back(std::move(new_buffer));
}

DevicePtr VkRuntime::get_snode_tree_device_ptr(int tree_id) {
  return root_buffers_[tree_id]->get_ptr();
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
