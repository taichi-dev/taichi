#include "taichi/backends/metal/kernel_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <random>
#include <string_view>

#include "taichi/backends/metal/constants.h"
#include "taichi/inc/constants.h"
#include "taichi/math/arithmetic.h"
#include "taichi/util/action_recorder.h"
#include "taichi/python/print_buffer.h"
#include "taichi/util/file_sequence_writer.h"

#ifdef TI_PLATFORM_OSX
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>

#include "taichi/backends/metal/api.h"
#include "taichi/program/program.h"
#endif  // TI_PLATFORM_OSX

TLANG_NAMESPACE_BEGIN
namespace metal {

#ifdef TI_PLATFORM_OSX

namespace {
namespace shaders {
#include "taichi/backends/metal/shaders/print.metal.h"
#include "taichi/backends/metal/shaders/runtime_utils.metal.h"
}  // namespace shaders

using KernelTaskType = OffloadedStmt::TaskType;
using BufferEnum = KernelAttributes::Buffers;

inline int infer_msl_version(const TaichiKernelAttributes::UsedFeatures &f) {
  if (f.simdgroup) {
    // https://developer.apple.com/documentation/metal/mtllanguageversion/version2_1
    return 131073;
  }
  return kMslVersionNone;
}

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
    std::memset(ptr_, 0, size_);
  }
  // Move only
  BufferMemoryView(BufferMemoryView &&) = default;
  BufferMemoryView &operator=(BufferMemoryView &&) = default;
  BufferMemoryView(const BufferMemoryView &) = delete;
  BufferMemoryView &operator=(const BufferMemoryView &) = delete;

  inline size_t size() const {
    return size_;
  }
  inline void *ptr() const {
    return ptr_;
  }

 private:
  size_t size_;
  void *ptr_;
};

// MetalRuntime maintains a series of MTLBuffers that are shared across all the
// Metal kernels mapped by a single Taichi kernel. This map stores those buffers
// from their enum. Each CompiledMtlKernelBase can then decide which specific
// buffers they will need to use in a launch.
using InputBuffersMap = std::unordered_map<BufferEnum, MTLBuffer *>;

// Info for launching a compiled Metal kernel
class CompiledMtlKernelBase {
 public:
  struct Params {
    bool is_jit_evaluator;
    const CompileConfig *config;
    const KernelAttributes *kernel_attribs;
    MTLDevice *device;
    MTLFunction *mtl_func;
  };

  explicit CompiledMtlKernelBase(Params &params)
      : kernel_attribs_(*params.kernel_attribs),
        config_(params.config),
        is_jit_evalutor_(params.is_jit_evaluator),
        pipeline_state_(
            new_compute_pipeline_state_with_function(params.device,
                                                     params.mtl_func)) {
    TI_ASSERT(pipeline_state_ != nullptr);
  }

  virtual ~CompiledMtlKernelBase() = default;

  inline KernelAttributes *kernel_attribs() {
    return &kernel_attribs_;
  }

  virtual void launch(InputBuffersMap &input_buffers,
                      MTLCommandBuffer *command_buffer) = 0;

 protected:
  using BindBuffers = std::vector<std::pair<MTLBuffer *, BufferEnum>>;

  void launch_if_not_empty(BindBuffers buffers,
                           MTLCommandBuffer *command_buffer) {
    const int num_threads = kernel_attribs_.num_threads;
    if (num_threads == 0) {
      return;
    }
    TI_ASSERT(buffers.size() == kernel_attribs_.buffers.size());
    auto encoder = new_compute_command_encoder(command_buffer);
    TI_ASSERT(encoder != nullptr);

    set_label(encoder.get(), kernel_attribs_.name);
    set_compute_pipeline_state(encoder.get(), pipeline_state_.get());

    for (int bi = 0; bi < buffers.size(); ++bi) {
      auto &b = buffers[bi];
      TI_ASSERT(b.second == kernel_attribs_.buffers[bi]);
      set_mtl_buffer(encoder.get(), b.first, /*offset=*/0, bi);
    }

    const auto tgs = get_thread_grid_settings(num_threads);
    if (!is_jit_evalutor_) {
      ActionRecorder::get_instance().record(
          "launch_kernel",
          {ActionArg("kernel_name", kernel_attribs_.name),
           ActionArg("num_threadgroups", tgs.num_threadgroups),
           ActionArg("num_threads_per_group", tgs.num_threads_per_group)});
    }

    dispatch_threadgroups(encoder.get(), tgs.num_threadgroups,
                          tgs.num_threads_per_group);
    end_encoding(encoder.get());
  }

  struct ThreadGridSettings {
    int num_threads_per_group;
    int num_threadgroups;
  };

  ThreadGridSettings get_thread_grid_settings(int num_threads) {
    int num_threads_per_group =
        get_max_total_threads_per_threadgroup(pipeline_state_.get());
    // Sometimes it is helpful to limit the maximum GPU block dim for the
    // kernels. E.g., when you are generating iPhone shaders on a Mac.
    const int prescribed_block_dim = config_->max_block_dim;
    if (prescribed_block_dim > 0) {
      num_threads_per_group =
          std::min(num_threads_per_group, prescribed_block_dim);
    }
    // Cap by |num_threads| in case this is a very small kernel.
    num_threads_per_group = std::min(num_threads_per_group, num_threads);

    int num_threadgroups =
        ((num_threads + num_threads_per_group - 1) / num_threads_per_group);
    // TODO(k-ye): Make sure |saturating_grid_dim| is configurable in ti.init()
    // before enabling this.
    // const int prescribed_grid_dim = config_->saturating_grid_dim;
    // if (prescribed_grid_dim > 0) {
    //   num_threadgroups = std::min(num_threadgroups, prescribed_grid_dim);
    // }
    return {num_threads_per_group, num_threadgroups};
  }

  KernelAttributes kernel_attribs_;
  const CompileConfig *const config_;
  const bool is_jit_evalutor_;
  nsobj_unique_ptr<MTLComputePipelineState> pipeline_state_;
};

// Metal kernel derived from a user Taichi kernel
class UserMtlKernel : public CompiledMtlKernelBase {
 public:
  using CompiledMtlKernelBase::CompiledMtlKernelBase;
  void launch(InputBuffersMap &input_buffers,
              MTLCommandBuffer *command_buffer) override {
    // 0 is valid for |num_threads|!
    TI_ASSERT(kernel_attribs_.num_threads >= 0);
    BindBuffers buffers;
    for (const auto b : kernel_attribs_.buffers) {
      buffers.push_back({input_buffers.find(b)->second, b});
    }
    launch_if_not_empty(std::move(buffers), command_buffer);
  }
};

// Internal Metal kernel used to maintain the kernel runtime data
class RuntimeListOpsMtlKernel : public CompiledMtlKernelBase {
 public:
  struct Params : public CompiledMtlKernelBase::Params {
    MemoryPool *mem_pool = nullptr;
    const SNodeDescriptorsMap *snode_descriptors = nullptr;

    const SNode *snode() const {
      return kernel_attribs->runtime_list_op_attribs->snode;
    }
  };

  explicit RuntimeListOpsMtlKernel(Params &params)
      : CompiledMtlKernelBase(params),
        parent_snode_id_(params.snode()->parent->id),
        child_snode_id_(params.snode()->id),
        args_mem_(std::make_unique<BufferMemoryView>(
            /*size=*/sizeof(int32_t) * 3,
            params.mem_pool)),
        args_buffer_(new_mtl_buffer_no_copy(params.device,
                                            args_mem_->ptr(),
                                            args_mem_->size())) {
    TI_ASSERT(args_buffer_ != nullptr);
    auto *mem = reinterpret_cast<int32_t *>(args_mem_->ptr());
    mem[0] = parent_snode_id_;
    mem[1] = child_snode_id_;
    const auto &sn_descs = *params.snode_descriptors;
    mem[2] = total_num_self_from_root(sn_descs, child_snode_id_);
    TI_DEBUG(
        "Registered RuntimeListOpsMtlKernel: name={} num_threads={} "
        "parent_snode={} "
        "child_snode={} max_num_elems={} ",
        params.kernel_attribs->name, params.kernel_attribs->num_threads, mem[0],
        mem[1], mem[2]);
    did_modify_range(args_buffer_.get(), /*location=*/0, args_mem_->size());
  }

  void launch(InputBuffersMap &input_buffers,
              MTLCommandBuffer *command_buffer) override {
    BindBuffers buffers;
    for (const auto b : kernel_attribs_.buffers) {
      if (b == BufferEnum::Context) {
        buffers.push_back({args_buffer_.get(), b});
      } else {
        buffers.push_back({input_buffers.find(b)->second, b});
      }
    }
    launch_if_not_empty(std::move(buffers), command_buffer);
  }

 private:
  const int parent_snode_id_;
  const int child_snode_id_;
  // For such Metal kernels, it always takes in an args buffer of two int32's:
  // args[0] = parent_snode_id
  // args[1] = child_snode_id
  // args[2] = child_snode.total_num_self_from_root
  // Note that this args buffer has nothing to do with the one passed to Taichi
  // kernel.
  // See taichi/backends/metal/shaders/runtime_kernels.metal.h
  std::unique_ptr<BufferMemoryView> args_mem_;
  nsobj_unique_ptr<MTLBuffer> args_buffer_;
};

// Info for launching a compiled Taichi kernel, which consists of a series of
// compiled Metal kernels.
class CompiledTaichiKernel {
 public:
  struct Params {
    std::string mtl_source_code;
    const TaichiKernelAttributes *ti_kernel_attribs;
    const KernelContextAttributes *ctx_attribs;
    const SNodeDescriptorsMap *snode_descriptors;
    MTLDevice *device;
    MemoryPool *mem_pool;
    KernelProfilerBase *profiler;
    const CompileConfig *compile_config;
  };

  CompiledTaichiKernel(Params params)
      : ti_kernel_attribs(*params.ti_kernel_attribs),
        ctx_attribs(*params.ctx_attribs) {
    auto *const device = params.device;
    auto kernel_lib = new_library_with_source(
        device, params.mtl_source_code, params.compile_config->fast_math,
        infer_msl_version(params.ti_kernel_attribs->used_features));
    if (kernel_lib == nullptr) {
      TI_ERROR("Failed to compile Metal kernel! Generated code:\n\n{}",
               params.mtl_source_code);
    }
    if (!ti_kernel_attribs.is_jit_evaluator &&
        ActionRecorder::get_instance().is_recording()) {
      static FileSequenceWriter writer("shader{:04d}.mtl", "Metal shader");
      auto fn = writer.write(params.mtl_source_code);
      ActionRecorder::get_instance().record(
          "save_kernel",
          {ActionArg("kernel_name", std::string(ti_kernel_attribs.name)),
           ActionArg("filename", fn)});
    }
    for (const auto &ka : ti_kernel_attribs.mtl_kernels_attribs) {
      auto mtl_func = new_function_with_name(kernel_lib.get(), ka.name);
      TI_ASSERT(mtl_func != nullptr);
      // Note that CompiledMtlKernel doesn't own |kernel_func|.
      std::unique_ptr<CompiledMtlKernelBase> kernel = nullptr;
      const auto ktype = ka.task_type;
      if (ktype == KernelTaskType::clear_list ||
          ktype == KernelTaskType::listgen) {
        RuntimeListOpsMtlKernel::Params kparams;
        kparams.kernel_attribs = &ka;
        kparams.is_jit_evaluator = ti_kernel_attribs.is_jit_evaluator;
        kparams.config = params.compile_config;
        kparams.device = device;
        kparams.mtl_func = mtl_func.get();
        kparams.mem_pool = params.mem_pool;
        kparams.snode_descriptors = params.snode_descriptors;
        kernel = std::make_unique<RuntimeListOpsMtlKernel>(kparams);
      } else {
        UserMtlKernel::Params kparams;
        kparams.kernel_attribs = &ka;
        kparams.is_jit_evaluator = ti_kernel_attribs.is_jit_evaluator;
        kparams.config = params.compile_config;
        kparams.device = device;
        kparams.mtl_func = mtl_func.get();
        kernel = std::make_unique<UserMtlKernel>(kparams);
      }

      TI_ASSERT(kernel != nullptr);
      compiled_mtl_kernels.push_back(std::move(kernel));
      TI_DEBUG("Added {} for Taichi kernel {}", ka.debug_string(),
               ti_kernel_attribs.name);
    }
    if (!ctx_attribs.empty()) {
      ctx_mem = std::make_unique<BufferMemoryView>(ctx_attribs.total_bytes(),
                                                   params.mem_pool);
      if (!ti_kernel_attribs.is_jit_evaluator) {
        ActionRecorder::get_instance().record(
            "allocate_context_buffer",
            {ActionArg("kernel_name", std::string(ti_kernel_attribs.name)),
             ActionArg("size_in_bytes", (int64)ctx_attribs.total_bytes())});
      }
      ctx_buffer =
          new_mtl_buffer_no_copy(device, ctx_mem->ptr(), ctx_mem->size());
    }
  }

  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  std::vector<std::unique_ptr<CompiledMtlKernelBase>> compiled_mtl_kernels;
  TaichiKernelAttributes ti_kernel_attribs;
  KernelContextAttributes ctx_attribs;
  std::unique_ptr<BufferMemoryView> ctx_mem;
  nsobj_unique_ptr<MTLBuffer> ctx_buffer;
};

class HostMetalCtxBlitter {
 public:
  HostMetalCtxBlitter(const CompiledTaichiKernel &kernel,
                      Context *host_ctx,
                      const std::string &kernel_name)
      : ti_kernel_attribs_(&kernel.ti_kernel_attribs),
        ctx_attribs_(&kernel.ctx_attribs),
        host_ctx_(host_ctx),
        kernel_ctx_mem_(kernel.ctx_mem.get()),
        kernel_ctx_buffer_(kernel.ctx_buffer.get()),
        kernel_name_(kernel_name) {
  }

  inline MTLBuffer *ctx_buffer() {
    return kernel_ctx_buffer_;
  }

  void host_to_metal() {
#define TO_METAL(type)                  \
  auto d = host_ctx_->get_arg<type>(i); \
  std::memcpy(device_ptr, &d, sizeof(d))

    if (ctx_attribs_->empty()) {
      return;
    }
    char *const base = (char *)kernel_ctx_mem_->ptr();
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      const auto dt = arg.dt;
      char *device_ptr = base + arg.offset_in_mem;
      if (!ti_kernel_attribs_->is_jit_evaluator) {
        ActionRecorder::get_instance().record(
            "context_host_to_metal",
            {ActionArg("kernel_name", kernel_name_), ActionArg("arg_id", i),
             ActionArg("offset_in_bytes", (int64)arg.offset_in_mem)});
      }
      if (arg.is_array) {
        const void *host_ptr = host_ctx_->get_arg<void *>(i);
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
    char *device_ptr = base + ctx_attribs_->ctx_bytes();
    std::memcpy(device_ptr, host_ctx_->extra_args,
                ctx_attribs_->extra_args_bytes());
#undef TO_METAL
    did_modify_range(kernel_ctx_buffer_, /*location=*/0,
                     kernel_ctx_mem_->size());
  }

  void metal_to_host() {
#define TO_HOST(type)                                   \
  const type d = *reinterpret_cast<type *>(device_ptr); \
  host_ctx_->set_arg<type>(i, d)

    if (ctx_attribs_->empty()) {
      return;
    }
    char *const base = (char *)kernel_ctx_mem_->ptr();
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      char *device_ptr = base + arg.offset_in_mem;
      if (arg.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);

        if (!ti_kernel_attribs_->is_jit_evaluator) {
          ActionRecorder::get_instance().record(
              "context_metal_to_host",
              {
                  ActionArg("kernel_name", kernel_name_),
                  ActionArg("arg_id", i),
                  ActionArg("size_in_bytes", (int64)arg.stride),
                  ActionArg("host_address",
                            fmt::format("0x{:x}", (uint64)host_ptr)),
                  ActionArg("device_address",
                            fmt::format("0x{:x}", (uint64)device_ptr)),
              });
        }
      }
    }
    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = base + ret.offset_in_mem;
      if (ret.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, ret.stride);
      } else {
        const auto dt = ret.dt;
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
          TI_ERROR("Metal does not support return value type={}",
                   metal_data_type_name(ret.dt));
        }
      }
    }
#undef TO_HOST
  }

  static std::unique_ptr<HostMetalCtxBlitter> maybe_make(
      const CompiledTaichiKernel &kernel,
      Context *ctx,
      std::string name = "") {
    if (kernel.ctx_attribs.empty()) {
      return nullptr;
    }
    return std::make_unique<HostMetalCtxBlitter>(kernel, ctx, name);
  }

 private:
  const TaichiKernelAttributes *const ti_kernel_attribs_;
  const KernelContextAttributes *const ctx_attribs_;
  Context *const host_ctx_;
  BufferMemoryView *const kernel_ctx_mem_;
  MTLBuffer *const kernel_ctx_buffer_;
  std::string kernel_name_;
};

}  // namespace

class KernelManager::Impl {
 public:
  explicit Impl(Params params)
      : config_(params.config),
        compiled_structs_(params.compiled_structs),
        mem_pool_(params.mem_pool),
        profiler_(params.profiler),
        command_buffer_id_(0) {
    if (config_->debug) {
      TI_ASSERT(is_metal_api_available());
    }
    device_ = mtl_create_system_default_device();
    TI_ASSERT(device_ != nullptr);
    command_queue_ = new_command_queue(device_.get());
    TI_ASSERT(command_queue_ != nullptr);
    create_new_command_buffer();

    if (compiled_structs_.root_size > 0) {
      root_mem_ = std::make_unique<BufferMemoryView>(
          compiled_structs_.root_size, mem_pool_);
      root_buffer_ = new_mtl_buffer_no_copy(device_.get(), root_mem_->ptr(),
                                            root_mem_->size());
      TI_ASSERT(root_buffer_ != nullptr);
      TI_DEBUG("Metal root buffer size: {} bytes", root_mem_->size());
      ActionRecorder::get_instance().record(
          "allocate_root_buffer",
          {ActionArg("size_in_bytes", (int64)root_mem_->size())});
    }

    global_tmps_mem_ = std::make_unique<BufferMemoryView>(
        taichi_global_tmp_buffer_size, mem_pool_);

    ActionRecorder::get_instance().record(
        "allocate_global_tmp_buffer",
        {ActionArg("size_in_bytes", (int64)taichi_global_tmp_buffer_size)});

    global_tmps_buffer_ = new_mtl_buffer_no_copy(
        device_.get(), global_tmps_mem_->ptr(), global_tmps_mem_->size());
    TI_ASSERT(global_tmps_buffer_ != nullptr);

    TI_ASSERT(compiled_structs_.runtime_size > 0);
    const int mem_pool_bytes = (config_->device_memory_GB * 1024 * 1024 * 1024);
    runtime_mem_ = std::make_unique<BufferMemoryView>(
        compiled_structs_.runtime_size + mem_pool_bytes, mem_pool_);
    runtime_buffer_ = new_mtl_buffer_no_copy(device_.get(), runtime_mem_->ptr(),
                                             runtime_mem_->size());
    TI_DEBUG(
        "Metal runtime buffer size: {} bytes (sizeof(Runtime)={} "
        "memory_pool={})",
        runtime_mem_->size(), compiled_structs_.runtime_size, mem_pool_bytes);

    ActionRecorder::get_instance().record(
        "allocate_runtime_buffer",
        {ActionArg("runtime_buffer_size_in_bytes", (int64)runtime_mem_->size()),
         ActionArg("runtime_struct_size_in_bytes",
                   (int64)compiled_structs_.runtime_size),
         ActionArg("memory_pool_size", (int64)mem_pool_bytes)});

    TI_ASSERT_INFO(
        runtime_buffer_ != nullptr,
        "Failed to allocate Metal runtime buffer, requested {} bytes",
        runtime_mem_->size());
    print_mem_ = std::make_unique<BufferMemoryView>(
        sizeof(shaders::PrintMsgAllocator) + shaders::kMetalPrintBufferSize,
        mem_pool_);
    print_buffer_ = new_mtl_buffer_no_copy(device_.get(), print_mem_->ptr(),
                                           print_mem_->size());
    TI_ASSERT(print_buffer_ != nullptr);

    init_runtime(params.root_id);
    init_print_buffer();
  }

  void register_taichi_kernel(const std::string &taichi_kernel_name,
                              const std::string &mtl_kernel_source_code,
                              const TaichiKernelAttributes &ti_kernel_attribs,
                              const KernelContextAttributes &ctx_attribs) {
    TI_ASSERT(compiled_taichi_kernels_.find(taichi_kernel_name) ==
              compiled_taichi_kernels_.end());

    if (config_->print_kernel_llvm_ir) {
      // If users have enabled |print_kernel_llvm_ir|, it probably means that
      // they want to see the compiled code on the given arch. Maybe rename this
      // flag, or add another flag (e.g. |print_kernel_source_code|)?
      TI_INFO("Metal source code for kernel <{}>\n{}", taichi_kernel_name,
              mtl_kernel_source_code);
    }
    CompiledTaichiKernel::Params params;
    params.mtl_source_code = mtl_kernel_source_code;
    params.ti_kernel_attribs = &ti_kernel_attribs;
    params.ctx_attribs = &ctx_attribs;
    params.snode_descriptors = &compiled_structs_.snode_descriptors;
    params.device = device_.get();
    params.mem_pool = mem_pool_;
    params.profiler = profiler_;
    params.compile_config = config_;
    compiled_taichi_kernels_[taichi_kernel_name] =
        std::make_unique<CompiledTaichiKernel>(params);
    TI_DEBUG("Registered Taichi kernel <{}>", taichi_kernel_name);
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            Context *ctx) {
    auto &ctk = *compiled_taichi_kernels_.find(taichi_kernel_name)->second;
    auto ctx_blitter =
        HostMetalCtxBlitter::maybe_make(ctk, ctx, taichi_kernel_name);
    if (config_->verbose_kernel_launches) {
      TI_INFO("Launching Taichi kernel <{}>", taichi_kernel_name);
    }

    InputBuffersMap input_buffers = {
        {BufferEnum::Root, root_buffer_.get()},
        {BufferEnum::GlobalTmps, global_tmps_buffer_.get()},
        {BufferEnum::Runtime, runtime_buffer_.get()},
        {BufferEnum::Print, print_buffer_.get()},
    };
    if (ctx_blitter) {
      ctx_blitter->host_to_metal();
      input_buffers[BufferEnum::Context] = ctk.ctx_buffer.get();
    }

    for (const auto &mk : ctk.compiled_mtl_kernels) {
      mk->launch(input_buffers, cur_command_buffer_.get());
    }
    const bool used_print = ctk.ti_kernel_attribs.used_features.print;
    if (ctx_blitter || used_print) {
      // TODO(k-ye): One optimization is to synchronize only when we absolutely
      // need to transfer the data back to host. This includes the cases where
      // an arg is 1) an array, or 2) used as return value.
      std::vector<MTLBuffer *> buffers_to_blit;
      if (ctx_blitter) {
        buffers_to_blit.push_back(ctx_blitter->ctx_buffer());
      }
      if (used_print) {
        buffers_to_blit.push_back(print_buffer_.get());
      }
      blit_buffers_and_sync(buffers_to_blit);

      if (ctx_blitter) {
        ctx_blitter->metal_to_host();
      }
      if (used_print) {
        flush_print_buffers();
      }
    }
  }

  void synchronize() {
    blit_buffers_and_sync();
  }

  PrintStringTable *print_strtable() {
    return &print_strtable_;
  }

 private:
  void init_runtime(int root_id) {
    using namespace shaders;
    char *addr = reinterpret_cast<char *>(runtime_mem_->ptr());
    const char *const addr_begin = addr;
    const int max_snodes = compiled_structs_.max_snodes;
    const auto &snode_descriptors = compiled_structs_.snode_descriptors;
    // init snode_metas
    for (int i = 0; i < max_snodes; ++i) {
      auto iter = snode_descriptors.find(i);
      if (iter == snode_descriptors.end()) {
        continue;
      }
      const SNodeDescriptor &sn_meta = iter->second;
      SNodeMeta *rtm_meta = reinterpret_cast<SNodeMeta *>(addr) + i;
      rtm_meta->element_stride = sn_meta.element_stride;
      rtm_meta->num_slots = sn_meta.num_slots;
      rtm_meta->mem_offset_in_parent = sn_meta.mem_offset_in_parent;
      switch (sn_meta.snode->type) {
        case SNodeType::dense:
          rtm_meta->type = SNodeMeta::Dense;
          break;
        case SNodeType::root:
          rtm_meta->type = SNodeMeta::Root;
          break;
        case SNodeType::bitmasked:
          rtm_meta->type = SNodeMeta::Bitmasked;
          break;
        case SNodeType::dynamic:
          rtm_meta->type = SNodeMeta::Dynamic;
          break;
        default:
          TI_ERROR("Unsupported SNode type={}",
                   snode_type_name(sn_meta.snode->type));
          break;
      }
      TI_DEBUG(
          "SnodeMeta\n  id={}\n  type={}\n  element_stride={}\n  "
          "num_slots={}\n",
          i, snode_type_name(sn_meta.snode->type), rtm_meta->element_stride,
          rtm_meta->num_slots);
    }
    size_t addr_offset = sizeof(SNodeMeta) * max_snodes;
    addr += addr_offset;
    TI_DEBUG("Initialized SNodeMeta, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    // init snode_extractors
    for (int i = 0; i < max_snodes; ++i) {
      auto iter = snode_descriptors.find(i);
      if (iter == snode_descriptors.end()) {
        continue;
      }
      const auto *sn = iter->second.snode;
      SNodeExtractors *rtm_ext = reinterpret_cast<SNodeExtractors *>(addr) + i;
      TI_DEBUG("SNodeExtractors snode={}", i);
      for (int j = 0; j < taichi_max_num_indices; ++j) {
        const auto &ext = sn->extractors[j];
        rtm_ext->extractors[j].start = ext.start;
        rtm_ext->extractors[j].num_bits = ext.num_bits;
        rtm_ext->extractors[j].acc_offset = ext.acc_offset;
        rtm_ext->extractors[j].num_elements = ext.num_elements;
        TI_DEBUG("  [{}] start={} num_bits={} acc_offset={} num_elements={}", j,
                 ext.start, ext.num_bits, ext.acc_offset, ext.num_elements);
      }
      TI_DEBUG("");
    }
    addr_offset = sizeof(SNodeExtractors) * max_snodes;
    addr += addr_offset;
    TI_DEBUG("Initialized SNodeExtractors, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    // init snode_lists
    ListManagerData *const rtm_list_begin =
        reinterpret_cast<ListManagerData *>(addr);
    for (int i = 0; i < max_snodes; ++i) {
      auto iter = snode_descriptors.find(i);
      if (iter == snode_descriptors.end()) {
        continue;
      }
      const SNodeDescriptor &sn_desc = iter->second;
      ListManagerData *rtm_list = reinterpret_cast<ListManagerData *>(addr) + i;
      rtm_list->element_stride = sizeof(ListgenElement);

      const int num_elems_per_chunk = compute_num_elems_per_chunk(
          sn_desc.total_num_self_from_root(snode_descriptors));
      rtm_list->log2_num_elems_per_chunk = log2int(num_elems_per_chunk);
      rtm_list->next = 0;
      TI_DEBUG("ListManagerData\n  id={}\n  num_elems_per_chunk={}\n", i,
               num_elems_per_chunk);
    }
    addr_offset = sizeof(ListManagerData) * max_snodes;
    addr += addr_offset;
    TI_DEBUG("Initialized ListManagerData, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    // init rand_seeds
    // TODO(k-ye): Provide a way to use a fixed seed in dev mode.
    std::mt19937 generator(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    std::uniform_int_distribution<uint32_t> distr(
        0, std::numeric_limits<uint32_t>::max());
    for (int i = 0; i < kNumRandSeeds; ++i) {
      uint32_t *s = reinterpret_cast<uint32_t *>(addr);
      *s = distr(generator);
      addr += sizeof(uint32_t);
    }
    TI_DEBUG("Initialized random seeds, size={} accumuated={}",
             kNumRandSeeds * sizeof(uint32_t), (addr - addr_begin));

    if (compiled_structs_.need_snode_lists_data) {
      auto *mem_alloc = reinterpret_cast<MemoryAllocator *>(addr);
      // Make sure the retured memory address is always greater than 1.
      mem_alloc->next = shaders::kAlignment;
      // root list data are static
      ListgenElement root_elem;
      root_elem.root_mem_offset = 0;
      for (int i = 0; i < taichi_max_num_indices; ++i) {
        root_elem.coords[i] = 0;
      }
      ListManager root_lm;
      root_lm.lm_data = rtm_list_begin + root_id;
      root_lm.mem_alloc = mem_alloc;
      root_lm.append(root_elem);
    }

    did_modify_range(runtime_buffer_.get(), /*location=*/0,
                     runtime_mem_->size());
  }

  void init_print_buffer() {
    // TODO(k-ye): Do we need this at all?
    did_modify_range(print_buffer_.get(), /*location=*/0, print_mem_->size());
  }

  void blit_buffers_and_sync(
      const std::vector<MTLBuffer *> &buffers_to_blit = {}) {
    // Blit the Metal buffers because they are in .managed mode.
    // We don't have to blit any of the root, global tmps or runtime buffer.
    // The data in these buffers are purely used inside GPU. When we need to
    // read back data from root buffer to CPU, it's done through that kernel's
    // context buffer.
    if (!buffers_to_blit.empty()) {
      auto encoder = new_blit_command_encoder(cur_command_buffer_.get());
      for (auto *b : buffers_to_blit) {
        synchronize_resource(encoder.get(), b);
      }
      end_encoding(encoder.get());
    }
    // Sync
    profiler_->start("metal_synchronize");
    commit_command_buffer(cur_command_buffer_.get());
    wait_until_completed(cur_command_buffer_.get());
    create_new_command_buffer();
    profiler_->stop();
  }

  void flush_print_buffers() {
    auto *pa =
        reinterpret_cast<shaders::PrintMsgAllocator *>(print_mem_->ptr());
    const int used_sz = std::min(pa->next, shaders::kMetalPrintBufferSize);
    using MsgType = shaders::PrintMsg::Type;
    char *buf = reinterpret_cast<char *>(pa + 1);
    const char *buf_end = buf + used_sz;

    while (buf < buf_end) {
      int32_t *msg_ptr = reinterpret_cast<int32_t *>(buf);
      const int num_entries = *msg_ptr;
      ++msg_ptr;
      shaders::PrintMsg msg(msg_ptr, num_entries);
      for (int i = 0; i < num_entries; ++i) {
        const auto dt = msg.pm_get_type(i);
        const int32_t x = msg.pm_get_data(i);
        if (dt == MsgType::I32) {
          py_cout << x;
        } else if (dt == MsgType::F32) {
          py_cout << *reinterpret_cast<const float *>(&x);
        } else if (dt == MsgType::Str) {
          py_cout << print_strtable_.get(x);
        } else {
          TI_ERROR("Unexecpted data type={}", dt);
        }
      }
      buf += shaders::mtl_compute_print_msg_bytes(num_entries);
    }

    if (pa->next >= shaders::kMetalPrintBufferSize) {
      py_cout << "...(maximum print buffer reached)\n";
    }

    pa->next = 0;
  }

  static int compute_num_elems_per_chunk(int n) {
    const int lb =
        (n + shaders::kTaichiNumChunks - 1) / shaders::kTaichiNumChunks;
    int result = 1024;
    while (result < lb) {
      result <<= 1;
    }
    return result;
  }

  void create_new_command_buffer() {
    cur_command_buffer_ = new_command_buffer(command_queue_.get());
    TI_ASSERT(cur_command_buffer_ != nullptr);
    set_label(cur_command_buffer_.get(),
              fmt::format("command_buffer_{}", command_buffer_id_++));
  }

  template <typename T>
  inline T load_global_tmp(int offset) const {
    return *reinterpret_cast<const T *>((const char *)global_tmps_mem_->ptr() +
                                        offset);
  }

  CompileConfig *const config_;
  const CompiledStructs compiled_structs_;
  MemoryPool *const mem_pool_;
  KernelProfilerBase *const profiler_;
  nsobj_unique_ptr<MTLDevice> device_;
  nsobj_unique_ptr<MTLCommandQueue> command_queue_;
  nsobj_unique_ptr<MTLCommandBuffer> cur_command_buffer_;
  std::size_t command_buffer_id_;
  std::unique_ptr<BufferMemoryView> root_mem_;
  nsobj_unique_ptr<MTLBuffer> root_buffer_;
  std::unique_ptr<BufferMemoryView> global_tmps_mem_;
  nsobj_unique_ptr<MTLBuffer> global_tmps_buffer_;
  std::unique_ptr<BufferMemoryView> runtime_mem_;
  nsobj_unique_ptr<MTLBuffer> runtime_buffer_;
  std::unique_ptr<BufferMemoryView> print_mem_;
  nsobj_unique_ptr<MTLBuffer> print_buffer_;
  std::unordered_map<std::string, std::unique_ptr<CompiledTaichiKernel>>
      compiled_taichi_kernels_;
  PrintStringTable print_strtable_;
};

#else

class KernelManager::Impl {
 public:
  explicit Impl(Params) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void register_taichi_kernel(const std::string &taichi_kernel_name,
                              const std::string &mtl_kernel_source_code,
                              const TaichiKernelAttributes &ti_kernel_attribs,
                              const KernelContextAttributes &ctx_attribs) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            Context *ctx) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void synchronize() {
    TI_ERROR("Metal not supported on the current OS");
  }

  PrintStringTable *print_strtable() {
    TI_ERROR("Metal not supported on the current OS");
    return nullptr;
  }
};

#endif  // TI_PLATFORM_OSX

KernelManager::KernelManager(Params params)
    : impl_(std::make_unique<Impl>(std::move(params))) {
}

KernelManager::~KernelManager() {
}

void KernelManager::register_taichi_kernel(
    const std::string &taichi_kernel_name,
    const std::string &mtl_kernel_source_code,
    const TaichiKernelAttributes &ti_kernel_attribs,
    const KernelContextAttributes &ctx_attribs) {
  impl_->register_taichi_kernel(taichi_kernel_name, mtl_kernel_source_code,
                                ti_kernel_attribs, ctx_attribs);
}

void KernelManager::launch_taichi_kernel(const std::string &taichi_kernel_name,
                                         Context *ctx) {
  impl_->launch_taichi_kernel(taichi_kernel_name, ctx);
}

void KernelManager::synchronize() {
  impl_->synchronize();
}

PrintStringTable *KernelManager::print_strtable() {
  return impl_->print_strtable();
}

}  // namespace metal
TLANG_NAMESPACE_END
