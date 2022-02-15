#include "taichi/backends/vulkan/runtime.h"
#include "taichi/program/program.h"

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

  int get_micros() {
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
                           RuntimeContext *host_ctx,
                           Device *device,
                           uint64_t *host_result_buffer,
                           DeviceAllocation *device_args_buffer,
                           DeviceAllocation *device_ret_buffer)
      : ctx_attribs_(ctx_attribs),
        host_ctx_(host_ctx),
        host_result_buffer_(host_result_buffer),
        device_args_buffer_(device_args_buffer),
        device_ret_buffer_(device_ret_buffer),
        device_(device) {
  }

  void host_to_device(
      const std::unordered_map<int, DeviceAllocation> &ext_arrays,
      const std::unordered_map<int, size_t> &ext_arr_size) {
    if (!ctx_attribs_->has_args()) {
      return;
    }

    char *const device_base =
        reinterpret_cast<char *>(device_->map(*device_args_buffer_));

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
          if (!host_ctx_->is_device_allocation[i] && ext_arr_size.at(i)) {
            // Only need to blit ext arrs (host array)
            DeviceAllocation buffer = ext_arrays.at(i);
            char *const device_arr_ptr =
                reinterpret_cast<char *>(device_->map(buffer));
            const void *host_ptr = host_ctx_->get_arg<void *>(i);
            std::memcpy(device_arr_ptr, host_ptr, ext_arr_size.at(i));
            device_->unmap(buffer);
          }
          // Substitue in the device address if supported
          if (device_->get_cap(
                  DeviceCapability::spirv_has_physical_storage_buffer)) {
            uint64_t addr =
                device_->get_memory_physical_pointer(ext_arrays.at(i));
            reinterpret_cast<uint64 *>(device_ptr)[0] = addr;
          }
          // We should not process the rest
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

    device_->unmap(*device_args_buffer_);
#undef TO_DEVICE
  }

  bool device_to_host(
      CommandList *cmdlist,
      const std::unordered_map<int, DeviceAllocation> &ext_arrays,
      const std::unordered_map<int, size_t> &ext_arr_size) {
    if (ctx_attribs_->empty()) {
      return false;
    }

    bool require_sync = ctx_attribs_->rets().size() > 0;
    if (!require_sync) {
      for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
        const auto &arg = ctx_attribs_->args()[i];
        if (arg.is_array) {
          if (!host_ctx_->is_device_allocation[i] && ext_arr_size.at(i)) {
            require_sync = true;
          }
        }
      }
    }

    if (require_sync) {
      device_->get_compute_stream()->submit_synced(cmdlist);
    } else {
      return false;
    }

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      if (arg.is_array) {
        if (!host_ctx_->is_device_allocation[i] && ext_arr_size.at(i)) {
          // Only need to blit ext arrs (host array)
          DeviceAllocation buffer = ext_arrays.at(i);
          char *const device_arr_ptr =
              reinterpret_cast<char *>(device_->map(buffer));
          void *host_ptr = host_ctx_->get_arg<void *>(i);
          std::memcpy(host_ptr, device_arr_ptr, ext_arr_size.at(i));
          device_->unmap(buffer);
        }
      }
    }

    if (!ctx_attribs_->has_rets())
      return require_sync;

    char *const device_base =
        reinterpret_cast<char *>(device_->map(*device_ret_buffer_));

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

    device_->unmap(*device_ret_buffer_);

    return true;
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(
      const KernelContextAttributes *ctx_attribs,
      RuntimeContext *host_ctx,
      Device *device,
      uint64_t *host_result_buffer,
      DeviceAllocation *device_args_buffer,
      DeviceAllocation *device_ret_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, host_ctx, device, host_result_buffer, device_args_buffer,
        device_ret_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  RuntimeContext *const host_ctx_;
  uint64_t *const host_result_buffer_;
  DeviceAllocation *const device_args_buffer_;
  DeviceAllocation *const device_ret_buffer_;
  Device *const device_;
};

}  // namespace

constexpr size_t kGtmpBufferSize = 1024 * 1024;
constexpr size_t kListGenBufferSize = 32 << 20;

// Info for launching a compiled Taichi kernel, which consists of a series of
// Vulkan pipelines.

CompiledTaichiKernel::CompiledTaichiKernel(const Params &ti_params)
    : ti_kernel_attribs_(*ti_params.ti_kernel_attribs),
      device_(ti_params.device) {
  input_buffers_[BufferType::GlobalTmps] = ti_params.global_tmps_buffer;
  input_buffers_[BufferType::ListGen] = ti_params.listgen_buffer;

  // Compiled_structs can be empty if loading a kernel from an AOT module as
  // the SNode are not re-compiled/structured. In thise case, we assume a
  // single root buffer size configured from the AOT module.
  if (ti_params.compiled_structs.empty() &&
      (ti_params.root_buffers.size() == 1)) {
    BufferInfo buffer = {BufferType::Root, 0};
    input_buffers_[buffer] = ti_params.root_buffers[0];
  } else {
    for (int root = 0; root < ti_params.compiled_structs.size(); ++root) {
      BufferInfo buffer = {BufferType::Root, root};
      input_buffers_[buffer] = ti_params.root_buffers[root];
    }
  }
  const auto arg_sz = ti_kernel_attribs_.ctx_attribs.args_bytes();
  const auto ret_sz = ti_kernel_attribs_.ctx_attribs.rets_bytes();

  args_buffer_size_ = arg_sz;
  ret_buffer_size_ = ret_sz;

  if (arg_sz) {
    args_buffer_size_ += ti_kernel_attribs_.ctx_attribs.extra_args_bytes();
  }

  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
  const auto &spirv_bins = ti_params.spirv_bins;
  TI_ASSERT(task_attribs.size() == spirv_bins.size());

  for (int i = 0; i < task_attribs.size(); ++i) {
    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary,
                                   (void *)spirv_bins[i].data(),
                                   spirv_bins[i].size() * sizeof(uint32_t)};
    auto vp =
        ti_params.device->create_pipeline(source_desc, task_attribs[i].name);
    pipelines_.push_back(std::move(vp));
  }
}

const TaichiKernelAttributes &CompiledTaichiKernel::ti_kernel_attribs() const {
  return ti_kernel_attribs_;
}

size_t CompiledTaichiKernel::num_pipelines() const {
  return pipelines_.size();
}

size_t CompiledTaichiKernel::get_args_buffer_size() const {
  return args_buffer_size_;
}

size_t CompiledTaichiKernel::get_ret_buffer_size() const {
  return ret_buffer_size_;
}

void CompiledTaichiKernel::generate_command_list(
    CommandList *cmdlist,
    DeviceAllocationGuard *args_buffer,
    DeviceAllocationGuard *ret_buffer,
    const std::unordered_map<int, DeviceAllocation> &ext_arrs) const {
  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;

  for (int i = 0; i < task_attribs.size(); ++i) {
    const auto &attribs = task_attribs[i];
    auto vp = pipelines_[i].get();
    const int group_x = (attribs.advisory_total_num_threads +
                         attribs.advisory_num_threads_per_group - 1) /
                        attribs.advisory_num_threads_per_group;
    ResourceBinder *binder = vp->resource_binder();
    for (auto &bind : attribs.buffer_binds) {
      if (bind.buffer.type == BufferType::ExtArr) {
        binder->rw_buffer(0, bind.binding, ext_arrs.at(bind.buffer.root_id));
      } else if (args_buffer && bind.buffer.type == BufferType::Args) {
        binder->buffer(0, bind.binding, *args_buffer);
      } else if (ret_buffer && bind.buffer.type == BufferType::Rets) {
        binder->rw_buffer(0, bind.binding, *ret_buffer);
      } else {
        DeviceAllocation *alloc = input_buffers_.at(bind.buffer);
        if (alloc) {
          binder->rw_buffer(0, bind.binding, *alloc);
        }
      }
    }

    if (attribs.task_type == OffloadedTaskType::listgen) {
      for (auto &bind : attribs.buffer_binds) {
        if (bind.buffer.type == BufferType::ListGen) {
          // FIXME: properlly support multiple list
          cmdlist->buffer_fill(input_buffers_.at(bind.buffer)->get_ptr(0),
                               kListGenBufferSize,
                               /*data=*/0);
          cmdlist->buffer_barrier(*input_buffers_.at(bind.buffer));
        }
      }
    }

    cmdlist->bind_pipeline(vp);
    cmdlist->bind_resources(binder);
    cmdlist->dispatch(group_x);
    cmdlist->memory_barrier();
  }
}

VkRuntime::VkRuntime(const Params &params)
    : device_(params.device), host_result_buffer_(params.host_result_buffer) {
  TI_ASSERT(host_result_buffer_ != nullptr);
  init_buffers();
}

VkRuntime::~VkRuntime() {
  synchronize();
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
  params.listgen_buffer = listgen_buffer_.get();

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

void VkRuntime::launch_kernel(KernelHandle handle, RuntimeContext *host_ctx) {
  auto *ti_kernel = ti_kernels_[handle.id_].get();

  std::unique_ptr<DeviceAllocationGuard> args_buffer{nullptr},
      ret_buffer{nullptr};

  if (ti_kernel->get_args_buffer_size()) {
    args_buffer = device_->allocate_memory_unique(
        {ti_kernel->get_args_buffer_size(),
         /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Uniform});
  }

  if (ti_kernel->get_ret_buffer_size()) {
    ret_buffer = device_->allocate_memory_unique(
        {ti_kernel->get_ret_buffer_size(),
         /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::Storage});
  }

  // Create context blitter
  auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
      &ti_kernel->ti_kernel_attribs().ctx_attribs, host_ctx, device_,
      host_result_buffer_, args_buffer.get(), ret_buffer.get());

  // `any_arrays` contain both external arrays and NDArrays
  std::unordered_map<int, DeviceAllocation> any_arrays;
  // `ext_array_size` only holds the size of external arrays (host arrays)
  // As buffer size information is only needed when it needs to be allocated
  // and transferred by the host
  std::unordered_map<int, size_t> ext_array_size;

  // Prepare context buffers & arrays
  if (ctx_blitter) {
    TI_ASSERT(ti_kernel->get_args_buffer_size() ||
              ti_kernel->get_ret_buffer_size());

    int i = 0;
    const auto &args = ti_kernel->ti_kernel_attribs().ctx_attribs.args();
    for (auto &arg : args) {
      if (arg.is_array) {
        if (host_ctx->is_device_allocation[i]) {
          // NDArray
          if (host_ctx->args[i]) {
            any_arrays[i] = *(DeviceAllocation *)(host_ctx->args[i]);
          } else {
            any_arrays[i] = kDeviceNullAllocation;
          }
        } else {
          // Compute ext arr sizes
          size_t size = arg.stride;
          bool has_zero_axis = false;

          for (int ax = 0; ax < 8; ax++) {
            // FIXME: how and when do we determine the size of ext arrs?
            size_t axis_size = host_ctx->extra_args[i][ax];
            if (axis_size) {
              if (has_zero_axis) {
                // e.g. shape [1, 0, 1]
                size = 0;
              } else {
                size *= host_ctx->extra_args[i][ax];
              }
            } else {
              has_zero_axis = true;
            }
          }

          ext_array_size[i] = size;

          // Alloc ext arr
          if (size) {
            DeviceAllocation extarr_buf = device_->allocate_memory(
                {size, /*host_write=*/true, /*host_read=*/true,
                 /*export_sharing=*/false, AllocUsage::Storage});
            any_arrays[i] = extarr_buf;
          } else {
            any_arrays[i] = kDeviceNullAllocation;
          }
        }
      }
      i++;
    }

    ctx_blitter->host_to_device(any_arrays, ext_array_size);
  }

  // Create new command list if current one is nullptr
  if (!current_cmdlist_) {
    ctx_buffers_.clear();
    current_cmdlist_ = device_->get_compute_stream()->new_command_list();
  }

  // Record commands
  ti_kernel->generate_command_list(current_cmdlist_.get(), args_buffer.get(),
                                   ret_buffer.get(), any_arrays);

  // Keep context buffers used in this dispatch
  if (ti_kernel->get_args_buffer_size()) {
    ctx_buffers_.push_back(std::move(args_buffer));
  }
  if (ti_kernel->get_ret_buffer_size()) {
    ctx_buffers_.push_back(std::move(ret_buffer));
  }

  // If we need to host sync, sync and remove in-flight references
  if (ctx_blitter) {
    if (ctx_blitter->device_to_host(current_cmdlist_.get(), any_arrays,
                                    ext_array_size)) {
      current_cmdlist_ = nullptr;
      ctx_buffers_.clear();
    }
  }

  // Dealloc external arrays
  for (auto pair : any_arrays) {
    if (pair.second != kDeviceNullAllocation) {
      if (!host_ctx->is_device_allocation[pair.first]) {
        device_->dealloc_memory(pair.second);
      }
    }
  }
}

void VkRuntime::synchronize() {
  if (current_cmdlist_) {
    device_->get_compute_stream()->submit(current_cmdlist_.get());
    current_cmdlist_ = nullptr;
  }
  device_->get_compute_stream()->command_sync();
  ctx_buffers_.clear();
}

Device *VkRuntime::get_ti_device() const {
  return device_;
}

void VkRuntime::init_buffers() {
  global_tmps_buffer_ = device_->allocate_memory_unique(
      {kGtmpBufferSize,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});

  listgen_buffer_ = device_->allocate_memory_unique(
      {kListGenBufferSize,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});

  // Need to zero fill the buffers, otherwise there could be NaN.
  Stream *stream = device_->get_compute_stream();
  auto cmdlist = stream->new_command_list();

  cmdlist->buffer_fill(global_tmps_buffer_->get_ptr(0), kGtmpBufferSize,
                       /*data=*/0);
  cmdlist->buffer_fill(listgen_buffer_->get_ptr(0), kListGenBufferSize,
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

VkRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    Device *device,
    const std::vector<CompiledSNodeStructs> &compiled_structs) {
  const auto id = Program::get_kernel_id();
  const auto taichi_kernel_name(fmt::format("{}_k{:04d}_vk", kernel->name, id));
  TI_TRACE("VK codegen for Taichi kernel={}", taichi_kernel_name);
  spirv::KernelCodegen::Params params;
  params.ti_kernel_name = taichi_kernel_name;
  params.kernel = kernel;
  params.compiled_structs = compiled_structs;
  params.device = device;
  params.enable_spv_opt =
      kernel->program->config.external_optimization_level > 0;
  spirv::KernelCodegen codegen(params);
  VkRuntime::RegisterParams res;
  codegen.run(res.kernel_attribs, res.task_spirv_source_codes);
  return std::move(res);
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
