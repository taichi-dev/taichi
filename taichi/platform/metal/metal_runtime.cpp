#include "metal_runtime.h"

#include <taichi/math/arithmetic.h>
#include <taichi/inc/constants.h>

#include <algorithm>
#include <cstring>
#include <string_view>

#define TI_RUNTIME_HOST
#include <taichi/runtime/llvm/context.h>
#undef TI_RUNTIME_HOST

#ifdef TI_PLATFORM_OSX
#include <sys/mman.h>
#include <unistd.h>

#include "metal_api.h"
#endif  // TI_PLATFORM_OSX

TLANG_NAMESPACE_BEGIN
namespace metal {

#ifdef TI_PLATFORM_OSX

namespace {
using KernelTaskType = OffloadedStmt::TaskType;

// This class requests the Metal buffer memory of |size| bytes from |mem_pool|.
// Once allocated, it does not own the memory (hence the name "view"). Instead,
// GC is deferred to the memory pool.
class BufferMemoryView {
 public:
  BufferMemoryView(size_t size, MemoryPool *mem_pool) {
    // Both |ptr_| and |size_| must be aligned to page size.
    size_ = iroundup(size, taichi_page_size);
    ptr_ = mem_pool->allocate(size_, /*alignment=*/taichi_page_size);
    TI_ASSERT(ptr_ != nullptr);
  }

  inline size_t size() const { return size_; }
  inline void *ptr() const { return ptr_; }

 private:
  size_t size_;
  void *ptr_;
};

struct MtlDataBuffers {
  MTLBuffer *root;
  MTLBuffer *global_tmps;
  MTLBuffer *args;  // nullable
};

// Info for launching a compiled Metal kernel
class CompiledMtlKernel {
 public:
  struct Params {
    const MetalKernelAttributes *kerenl_attribs;
    MTLDevice *device;
    MTLFunction *mtl_func;
    ProfilerBase *profiler;
  };

  CompiledMtlKernel(Params params)
      : kernel_attribs_(*params.kerenl_attribs),
        pipeline_state_(new_compute_pipeline_state_with_function(
            params.device, params.mtl_func)),
        profiler_(params.profiler),
        profiler_id_(fmt::format("{}_dispatch", kernel_attribs_.name)) {
    TI_ASSERT(pipeline_state_ != nullptr);
  }

  void launch(MtlDataBuffers data_buffers, MTLCommandBuffer *command_buffer) {
    // 0 is valid for |num_threads|!
    TI_ASSERT(kernel_attribs_.num_threads >= 0);
    launch_if_not_empty(std::move(data_buffers), command_buffer);
    if ((kernel_attribs_.task_type == KernelTaskType::range_for) &&
        !kernel_attribs_.range_for_attribs.const_range()) {
      // Set |num_thread| to an invalid number to make sure the next launch
      // re-computes it correctly.
      kernel_attribs_.num_threads = -1;
    }
  }

  inline MetalKernelAttributes *kernel_attribs() { return &kernel_attribs_; }

 private:
  void launch_if_not_empty(MtlDataBuffers data_buffers,
                           MTLCommandBuffer *command_buffer) {
    const int num_threads = kernel_attribs_.num_threads;
    if (num_threads == 0) {
      return;
    }
    profiler_->start(profiler_id_);
    auto encoder = new_compute_command_encoder(command_buffer);
    TI_ASSERT(encoder != nullptr);

    set_compute_pipeline_state(encoder.get(), pipeline_state_.get());
    int buffer_index = 0;
    if (data_buffers.root) {
      set_mtl_buffer(encoder.get(), data_buffers.root, /*offset=*/0,
                     buffer_index++);
    }
    set_mtl_buffer(encoder.get(), data_buffers.global_tmps, /*offset=*/0,
                   buffer_index++);
    if (data_buffers.args) {
      set_mtl_buffer(encoder.get(), data_buffers.args, /*offset=*/0,
                     buffer_index++);
    }
    const int num_threads_per_group =
        get_max_total_threads_per_threadgroup(pipeline_state_.get());
    const int num_groups =
        ((num_threads + num_threads_per_group - 1) / num_threads_per_group);
    dispatch_threadgroups(encoder.get(), num_groups,
                          std::min(num_threads, num_threads_per_group));
    end_encoding(encoder.get());
    profiler_->stop();
  }

  MetalKernelAttributes kernel_attribs_;
  nsobj_unique_ptr<MTLComputePipelineState> pipeline_state_{nullptr};
  ProfilerBase *const profiler_;
  const std::string profiler_id_;
};

// Info for launching a compiled Taichi kernel, which consists of a series of
// compiled Metal kernels.
class CompiledTaichiKernel {
 public:
  struct Params {
    std::string_view taichi_kernel_name;
    std::string_view mtl_source_code;
    const std::vector<MetalKernelAttributes> *mtl_kernels_attribs;
    const MetalKernelArgsAttributes *args_attribs;
    MTLDevice *device;
    MemoryPool *mem_pool;
    ProfilerBase *profiler;
  };

  CompiledTaichiKernel(Params params)
      : args_attribs(*params.args_attribs),
        mtl_source_code_(params.mtl_source_code),
        profiler_(params.profiler) {
    auto *const device = params.device;
    auto kernel_lib = new_library_with_source(device, mtl_source_code_);
    TI_ASSERT(kernel_lib != nullptr);
    for (const auto &ka : *(params.mtl_kernels_attribs)) {
      auto mtl_func = new_function_with_name(kernel_lib.get(), ka.name);
      TI_ASSERT(mtl_func != nullptr);
      // Note that CompiledMtlKernel doesn't own |kernel_func|.
      CompiledMtlKernel::Params params;
      params.kerenl_attribs = &ka;
      params.device = device;
      params.mtl_func = mtl_func.get();
      params.profiler = profiler_;
      compiled_mtl_kernels.push_back(
          std::make_unique<CompiledMtlKernel>(params));
    }
    if (args_attribs.has_args()) {
      args_mem = std::make_unique<BufferMemoryView>(args_attribs.total_bytes(),
                                                    params.mem_pool);
      args_buffer =
          new_mtl_buffer_no_copy(device, args_mem->ptr(), args_mem->size());
    }
  }

  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  std::vector<std::unique_ptr<CompiledMtlKernel>> compiled_mtl_kernels;
  MetalKernelArgsAttributes args_attribs;
  std::unique_ptr<BufferMemoryView> args_mem{nullptr};
  nsobj_unique_ptr<MTLBuffer> args_buffer{nullptr};

 private:
  std::string mtl_source_code_;
  ProfilerBase *const profiler_;
};

class HostMetalArgsBlitter {
 public:
  HostMetalArgsBlitter(const MetalKernelArgsAttributes *args_attribs,
                       Context *ctx, BufferMemoryView *args_mem)
      : args_attribs_(args_attribs), ctx_(ctx), args_mem_(args_mem) {}

  void host_to_metal() {
#define TO_METAL(type)             \
  auto d = ctx_->get_arg<type>(i); \
  std::memcpy(device_ptr, &d, sizeof(d))

    if (!args_attribs_->has_args()) {
      return;
    }
    char *const base = (char *)args_mem_->ptr();
    for (int i = 0; i < args_attribs_->args().size(); ++i) {
      const auto &arg = args_attribs_->args()[i];
      const auto dt = arg.dt;
      char *device_ptr = base + arg.offset_in_mem;
      if (arg.is_array) {
        const void *host_ptr = ctx_->get_arg<void *>(i);
        std::memcpy(device_ptr, host_ptr, arg.stride);
      } else if (dt == MetalDataType::i32) {
        TO_METAL(int32);
      } else if (dt == MetalDataType::u32) {
        TO_METAL(uint32);
      } else if (dt == MetalDataType::f32) {
        TO_METAL(float32);
      } else if (dt == MetalDataType::i8) {
        TO_METAL(int8);
      } else if (dt == MetalDataType::i16) {
        TO_METAL(int16);
      } else if (dt == MetalDataType::u8) {
        TO_METAL(uint8);
      } else if (dt == MetalDataType::u16) {
        TO_METAL(uint16);
      } else {
        TI_ERROR("Metal does not support arg type={}",
                 metal_data_type_name(arg.dt));
      }
    }
    char *device_ptr = base + args_attribs_->args_bytes();
    std::memcpy(device_ptr, ctx_->extra_args,
                args_attribs_->extra_args_bytes());
#undef TO_METAL
  }

  void metal_to_host() {
#define TO_HOST(type)                                   \
  const type d = *reinterpret_cast<type *>(device_ptr); \
  ctx_->set_arg<type>(i, d)

    if (!args_attribs_->has_args()) {
      return;
    }
    char *const base = (char *)args_mem_->ptr();
    for (int i = 0; i < args_attribs_->args().size(); ++i) {
      const auto &arg = args_attribs_->args()[i];
      char *device_ptr = base + arg.offset_in_mem;
      if (arg.is_array) {
        void *host_ptr = ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);
      } else if (arg.is_return_val) {
        const auto dt = arg.dt;
        if (dt == MetalDataType::i32) {
          TO_HOST(int32);
        } else if (dt == MetalDataType::u32) {
          TO_HOST(uint32);
        } else if (dt == MetalDataType::f32) {
          TO_HOST(float32);
        } else if (dt == MetalDataType::i8) {
          TO_HOST(int8);
        } else if (dt == MetalDataType::i16) {
          TO_HOST(int16);
        } else if (dt == MetalDataType::u8) {
          TO_HOST(uint8);
        } else if (dt == MetalDataType::u16) {
          TO_HOST(uint16);
        } else {
          TI_ERROR("Metal does not support arg type={}",
                   metal_data_type_name(arg.dt));
        }
      }
    }
#undef TO_HOST
  }

  static std::unique_ptr<HostMetalArgsBlitter> make_if_has_args(
      const CompiledTaichiKernel &kernel, Context *ctx) {
    if (!kernel.args_attribs.has_args()) {
      return nullptr;
    }
    return std::make_unique<HostMetalArgsBlitter>(&kernel.args_attribs, ctx,
                                                  kernel.args_mem.get());
  }

 private:
  const MetalKernelArgsAttributes *const args_attribs_;
  Context *const ctx_;
  BufferMemoryView *const args_mem_{nullptr};
};

}  // namespace

class MetalRuntime::Impl {
 public:
  explicit Impl(Params params)
      : config_(params.config),
        mem_pool_(params.mem_pool),
        profiler_(params.profiler),
        needs_root_buffer_(params.root_size > 0) {
    if (config_->debug) {
      TI_ASSERT(is_metal_api_available());
    }
    device_ = mtl_create_system_default_device();
    TI_ASSERT(device_ != nullptr);
    command_queue_ = new_command_queue(device_.get());
    TI_ASSERT(command_queue_ != nullptr);
    create_new_command_buffer();

    auto *llvm_ctx = params.llvm_ctx;
    auto *llvm_rtm = params.llvm_runtime;
    TI_ASSERT(llvm_ctx != nullptr && llvm_rtm != nullptr);
    const size_t rtm_root_mem_size = llvm_ctx->lookup_function<size_t(void *)>(
        "LLVMRuntime_get_root_mem_size")(llvm_rtm);
    if (rtm_root_mem_size > 0) {
      TI_ASSERT(needs_root_buffer_);
      // Make sure the runtime's root memory is large enough.
      TI_ASSERT(iroundup(params.root_size, taichi_page_size) <=
                rtm_root_mem_size);
      auto *rtm_root_mem = params.llvm_ctx->lookup_function<uint8 *(void *)>(
          "LLVMRuntime_get_root")(llvm_rtm);
      TI_ASSERT(rtm_root_mem != nullptr);
      root_buffer_ = new_mtl_buffer_no_copy(device_.get(), rtm_root_mem,
                                            rtm_root_mem_size);
      TI_ASSERT(root_buffer_ != nullptr);
      TI_DEBUG("Metal root buffer size: {} bytes", rtm_root_mem_size);
    } else {
      TI_ASSERT(!needs_root_buffer_);
      TI_DEBUG("Metal root buffer is empty");
    }

    // Make sure we don't have to round up global temporaries' buffer size.
    TI_ASSERT(iroundup(taichi_global_tmp_buffer_size, taichi_page_size) ==
              taichi_global_tmp_buffer_size);
    global_tmps_mem_begin_ = params.llvm_ctx->lookup_function<uint8 *(void *)>(
        "LLVMRuntime_get_temporaries")(llvm_rtm);
    global_tmps_buffer_ = new_mtl_buffer_no_copy(
        device_.get(), global_tmps_mem_begin_, taichi_global_tmp_buffer_size);
    TI_ASSERT(global_tmps_buffer_ != nullptr);
  }

  void register_taichi_kernel(
      const std::string &taichi_kernel_name,
      const std::string &mtl_kernel_source_code,
      const std::vector<MetalKernelAttributes> &kernels_attribs,
      size_t global_tmps_size, const MetalKernelArgsAttributes &args_attribs) {
    TI_ASSERT(compiled_taichi_kernels_.find(taichi_kernel_name) ==
              compiled_taichi_kernels_.end());
    TI_ASSERT(iroundup(global_tmps_size, taichi_page_size) <=
              taichi_global_tmp_buffer_size);

    if (config_->print_kernel_llvm_ir) {
      // If users have enabled |print_kernel_llvm_ir|, it probably means that
      // they want to see the compiled code on the given arch. Maybe rename this
      // flag, or add another flag (e.g. |print_kernel_source_code|)?
      TI_INFO("Metal source code for kernel <{}>\n{}", taichi_kernel_name,
              mtl_kernel_source_code);
    }
    CompiledTaichiKernel::Params params;
    params.taichi_kernel_name = taichi_kernel_name;
    params.mtl_source_code = mtl_kernel_source_code;
    params.mtl_kernels_attribs = &kernels_attribs;
    params.args_attribs = &args_attribs;
    params.device = device_.get();
    params.mem_pool = mem_pool_;
    params.profiler = profiler_;
    compiled_taichi_kernels_[taichi_kernel_name] =
        std::make_unique<CompiledTaichiKernel>(params);
    TI_INFO("Registered Taichi kernel <{}>", taichi_kernel_name);
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            Context *ctx) {
    auto &ctk = *compiled_taichi_kernels_.find(taichi_kernel_name)->second;
    auto args_blitter = HostMetalArgsBlitter::make_if_has_args(ctk, ctx);
    if (config_->verbose_kernel_launches) {
      TI_INFO("Lauching Taichi kernel <{}>", taichi_kernel_name);
    }
    if (args_blitter) {
      args_blitter->host_to_metal();
    }
    for (const auto &mk : ctk.compiled_mtl_kernels) {
      auto *ka = mk->kernel_attribs();
      if ((ka->task_type == KernelTaskType::range_for) &&
          !ka->range_for_attribs.const_range()) {
        // If the for loop range is determined at runtime, it will be computed
        // in the previous serial kernel. We need to read it back to the host
        // side to decide how many kernel threads to launch.
        synchronize();
        const int begin =
            ka->range_for_attribs.const_begin
                ? ka->range_for_attribs.begin
                : load_global_tmp<int>(ka->range_for_attribs.begin);
        const int end = ka->range_for_attribs.const_end
                            ? ka->range_for_attribs.end
                            : load_global_tmp<int>(ka->range_for_attribs.end);
        TI_ASSERT(ka->num_threads == -1);
        ka->num_threads = end - begin;
      }
      MtlDataBuffers data_buffers;
      data_buffers.root = root_buffer_.get();
      data_buffers.global_tmps = global_tmps_buffer_.get();
      data_buffers.args = ctk.args_buffer.get();
      mk->launch(std::move(data_buffers), cur_command_buffer_.get());
    }
    if (args_blitter) {
      // TODO(k-ye): One optimization is to synchronize only when we absolutely
      // need to transfer the data back to host. This includes the cases where
      // an arg is 1) an array, or 2) used as return value.
      synchronize();
      args_blitter->metal_to_host();
    }
  }

  void synchronize() {
    profiler_->start("metal_synchronize");
    commit_command_buffer(cur_command_buffer_.get());
    wait_until_completed(cur_command_buffer_.get());
    create_new_command_buffer();
    profiler_->stop();
  }

 private:
  void create_new_command_buffer() {
    cur_command_buffer_ = new_command_buffer(command_queue_.get());
    TI_ASSERT(cur_command_buffer_ != nullptr);
  }

  template <typename T>
  inline T load_global_tmp(int offset) const {
    return *reinterpret_cast<const T *>(global_tmps_mem_begin_ + offset);
  }

  CompileConfig *const config_;
  MemoryPool *const mem_pool_;
  ProfilerBase *const profiler_;
  const bool needs_root_buffer_;
  nsobj_unique_ptr<MTLDevice> device_{nullptr};
  nsobj_unique_ptr<MTLCommandQueue> command_queue_{nullptr};
  nsobj_unique_ptr<MTLCommandBuffer> cur_command_buffer_{nullptr};
  nsobj_unique_ptr<MTLBuffer> root_buffer_{nullptr};
  uint8_t *global_tmps_mem_begin_;
  nsobj_unique_ptr<MTLBuffer> global_tmps_buffer_{nullptr};
  std::unordered_map<std::string, std::unique_ptr<CompiledTaichiKernel>>
      compiled_taichi_kernels_;
};

#else

class MetalRuntime::Impl {
 public:
  explicit Impl(Params) { TI_ERROR("Metal not supported on the current OS"); }

  void register_taichi_kernel(
      const std::string &taichi_kernel_name,
      const std::string &mtl_kernel_source_code,
      const std::vector<MetalKernelAttributes> &kernels_attribs,
      size_t global_tmps_size, const MetalKernelArgsAttributes &args_attribs) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            Context *ctx) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void synchronize() { TI_ERROR("Metal not supported on the current OS"); }
};

#endif  // TI_PLATFORM_OSX

MetalRuntime::MetalRuntime(Params params)
    : impl_(std::make_unique<Impl>(std::move(params))) {}

MetalRuntime::~MetalRuntime() {}

void MetalRuntime::register_taichi_kernel(
    const std::string &taichi_kernel_name,
    const std::string &mtl_kernel_source_code,
    const std::vector<MetalKernelAttributes> &kernels_attribs,
    size_t global_tmps_size, const MetalKernelArgsAttributes &args_attribs) {
  impl_->register_taichi_kernel(taichi_kernel_name, mtl_kernel_source_code,
                                kernels_attribs, global_tmps_size,
                                args_attribs);
}

void MetalRuntime::launch_taichi_kernel(const std::string &taichi_kernel_name,
                                        Context *ctx) {
  impl_->launch_taichi_kernel(taichi_kernel_name, ctx);
}

void MetalRuntime::synchronize() { impl_->synchronize(); }

}  // namespace metal
TLANG_NAMESPACE_END
