#include "taichi/backends/metal/kernel_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string_view>

#include "taichi/backends/metal/constants.h"
#include "taichi/backends/metal/device.h"
#include "taichi/backends/metal/features.h"
#include "taichi/backends/metal/runtime_utils.h"
#include "taichi/inc/constants.h"
#include "taichi/math/arithmetic.h"
#include "taichi/program/kernel.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/util/action_recorder.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/util/str.h"
#include "taichi/common/exceptions.h"

#ifdef TI_PLATFORM_OSX
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>

#include "taichi/backends/metal/api.h"
#include "taichi/program/program.h"
#endif  // TI_PLATFORM_OSX

namespace taichi {
namespace lang {
namespace metal {

#ifdef TI_PLATFORM_OSX

namespace {
namespace shaders {
#include "taichi/backends/metal/shaders/print.metal.h"
#include "taichi/backends/metal/shaders/runtime_utils.metal.h"
}  // namespace shaders

using KernelTaskType = OffloadedTaskType;

inline int infer_msl_version(const TaichiKernelAttributes::UsedFeatures &f) {
  if (f.simdgroup) {
    // https://developer.apple.com/documentation/metal/mtllanguageversion/version2_1
    return 131073;
  }
  return kMslVersionNone;
}

// MetalRuntime maintains a series of MTLBuffers that are shared across all the
// Metal kernels mapped by a single Taichi kernel. This map stores those buffers
// from their enum. Each CompiledMtlKernelBase can then decide which specific
// buffers they will need to use in a launch.
using InputBuffersMap =
    std::unordered_map<BufferDescriptor, MTLBuffer *, BufferDescriptor::Hasher>;

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
  using BindBuffers = std::vector<std::pair<MTLBuffer *, BufferDescriptor>>;

  void launch_if_not_empty(BindBuffers buffers,
                           MTLCommandBuffer *command_buffer) {
    const int num_threads = kernel_attribs_.advisory_total_num_threads;
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

    const auto tgs = get_thread_grid_settings(
        num_threads, kernel_attribs_.advisory_num_threads_per_group);
    if (!is_jit_evalutor_) {
      const auto tt = kernel_attribs_.task_type;
      std::vector<ActionArg> record_args = {
          ActionArg("mtl_kernel_name", kernel_attribs_.name),
          ActionArg("advisory_num_threads", num_threads),
          ActionArg("num_threadgroups", tgs.num_threadgroups),
          ActionArg("num_threads_per_group", tgs.num_threads_per_group),
          ActionArg("task_type", offloaded_task_type_name(tt)),
      };
      const auto &buffers = kernel_attribs_.buffers;
      for (int i = 0; i < buffers.size(); ++i) {
        record_args.push_back(ActionArg(fmt::format("mtl_buffer_{}", i),
                                        buffers[i].debug_string()));
      }
      ActionRecorder::get_instance().record("launch_kernel",
                                            std::move(record_args));
    }
    TI_TRACE(
        "Dispatching Metal kernel {}, num_threadgroups={} "
        "num_threads_per_group={}",
        kernel_attribs_.name, tgs.num_threadgroups, tgs.num_threads_per_group);
    dispatch_threadgroups(encoder.get(), tgs.num_threadgroups,
                          tgs.num_threads_per_group);
    end_encoding(encoder.get());
  }

  struct ThreadGridSettings {
    int num_threads_per_group;
    int num_threadgroups;
  };

  ThreadGridSettings get_thread_grid_settings(
      int advisory_num_threads,
      int advisory_num_threads_per_group) {
    int num_threads_per_group =
        get_max_total_threads_per_threadgroup(pipeline_state_.get());
    // Sometimes it is helpful to limit the maximum GPU block dim for the
    // kernels. E.g., when you are generating iPhone shaders on a Mac.
    const int prescribed_block_dim = config_->max_block_dim;
    if (prescribed_block_dim > 0) {
      num_threads_per_group =
          std::min(num_threads_per_group, prescribed_block_dim);
    }
    if (advisory_num_threads_per_group > 0) {
      num_threads_per_group =
          std::min(num_threads_per_group, advisory_num_threads_per_group);
    }
    // Cap by |num_threads| in case this is a very small kernel.
    num_threads_per_group =
        std::min(num_threads_per_group, advisory_num_threads);

    int num_threadgroups = ((advisory_num_threads + num_threads_per_group - 1) /
                            num_threads_per_group);
    const int prescribed_grid_dim = config_->saturating_grid_dim;
    if (prescribed_grid_dim > 0) {
      num_threadgroups = std::min(num_threadgroups, prescribed_grid_dim);
    }
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
    TI_ASSERT(kernel_attribs_.advisory_total_num_threads >= 0);
    BindBuffers buffers;
    for (const auto &b : kernel_attribs_.buffers) {
      buffers.push_back({input_buffers.find(b)->second, b});
    }
    launch_if_not_empty(std::move(buffers), command_buffer);
  }
};

// Internal Metal kernel used to maintain the kernel runtime data
class SparseRuntimeMtlKernelBase : public CompiledMtlKernelBase {
 public:
  struct Params : public CompiledMtlKernelBase::Params {
    MemoryPool *mem_pool = nullptr;
  };

  explicit SparseRuntimeMtlKernelBase(Params &params, int args_size)
      : CompiledMtlKernelBase(params),
        args_mem_(
            std::make_unique<BufferMemoryView>(args_size, params.mem_pool)),
        args_buffer_(new_mtl_buffer_no_copy(params.device,
                                            args_mem_->ptr(),
                                            args_mem_->size())) {
    TI_ASSERT(args_buffer_ != nullptr);
  }

  void launch(InputBuffersMap &input_buffers,
              MTLCommandBuffer *command_buffer) override {
    BindBuffers buffers;
    for (const auto &b : kernel_attribs_.buffers) {
      if (b.type() == BufferDescriptor::Type::Context) {
        buffers.push_back({args_buffer_.get(), b});
      } else {
        buffers.push_back({input_buffers.find(b)->second, b});
      }
    }
    launch_if_not_empty(std::move(buffers), command_buffer);
  }

 protected:
  std::unique_ptr<BufferMemoryView> args_mem_;
  nsobj_unique_ptr<MTLBuffer> args_buffer_;
};

class ListgenOpMtlKernel : public SparseRuntimeMtlKernelBase {
 public:
  struct Params : public SparseRuntimeMtlKernelBase::Params {
    const SNode *snode() const {
      return kernel_attribs->runtime_list_op_attribs->snode;
    }
  };

  explicit ListgenOpMtlKernel(Params &params)
      : SparseRuntimeMtlKernelBase(params, /*args_size=*/sizeof(int32_t) * 2) {
    // For such Metal kernels, it always takes in an args buffer of 2 int32's:
    // args[0] = parent_snode_id
    // args[1] = child_snode_id
    // Note that this args buffer has nothing to do with the one passed to
    // Taichi kernel. See taichi/backends/metal/shaders/runtime_kernels.metal.h
    const int parent_snode_id = params.snode()->parent->id;
    const int child_snode_id = params.snode()->id;
    auto *mem = reinterpret_cast<int32_t *>(args_mem_->ptr());
    mem[0] = parent_snode_id;
    mem[1] = child_snode_id;
    TI_DEBUG(
        "Registered ListgenOpMtlKernel: name={} num_threads={} "
        "parent_snode={} "
        "child_snode={}",
        params.kernel_attribs->name,
        params.kernel_attribs->advisory_total_num_threads, mem[0], mem[1]);
    did_modify_range(args_buffer_.get(), /*location=*/0, args_mem_->size());
  }
};

class GcOpMtlKernel : public SparseRuntimeMtlKernelBase {
 public:
  struct Params : public SparseRuntimeMtlKernelBase::Params {
    const SNode *snode() const {
      return kernel_attribs->gc_op_attribs->snode;
    }
  };

  explicit GcOpMtlKernel(Params &params)
      : SparseRuntimeMtlKernelBase(params, /*args_size=*/sizeof(int32_t)) {
    const int snode_id = params.snode()->id;
    auto *mem = reinterpret_cast<int32_t *>(args_mem_->ptr());
    mem[0] = snode_id;
    TI_DEBUG("Registered GcOpMtlKernel: name={} num_threads={} snode_id={}",
             params.kernel_attribs->name,
             params.kernel_attribs->advisory_total_num_threads, mem[0]);
    did_modify_range(args_buffer_.get(), /*location=*/0, args_mem_->size());
  }
};

// Info for launching a compiled Taichi kernel, which consists of a series of
// compiled Metal kernels.
class CompiledTaichiKernel {
 public:
  struct Params {
    std::string mtl_source_code;
    const TaichiKernelAttributes *ti_kernel_attribs;
    const KernelContextAttributes *ctx_attribs;
    MTLDevice *device;
    MemoryPool *mem_pool;
    KernelProfilerBase *profiler;
    const CompileConfig *compile_config;
    const Kernel *kernel;
    Device *rhi_device;
  };

  CompiledTaichiKernel(Params params)
      : ti_kernel_attribs(*params.ti_kernel_attribs),
        ctx_attribs(*params.ctx_attribs),
        kernel_(params.kernel),
        rhi_device_(params.rhi_device) {
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
          {ActionArg("ti_kernel_name", std::string(ti_kernel_attribs.name)),
           ActionArg("filename", fn)});
    }
    for (const auto &ka : ti_kernel_attribs.mtl_kernels_attribs) {
      auto mtl_func = new_function_with_name(kernel_lib.get(), ka.name);
      TI_ASSERT(mtl_func != nullptr);
      // Note that CompiledMtlKernel doesn't own |kernel_func|.
      std::unique_ptr<CompiledMtlKernelBase> kernel = nullptr;
      const auto ktype = ka.task_type;
      if (ktype == KernelTaskType::listgen) {
        ListgenOpMtlKernel::Params kparams;
        kparams.kernel_attribs = &ka;
        kparams.is_jit_evaluator = false;
        kparams.config = params.compile_config;
        kparams.device = device;
        kparams.mtl_func = mtl_func.get();
        kparams.mem_pool = params.mem_pool;
        kernel = std::make_unique<ListgenOpMtlKernel>(kparams);
      } else if (ktype == KernelTaskType::gc) {
        GcOpMtlKernel::Params kparams;
        kparams.kernel_attribs = &ka;
        kparams.is_jit_evaluator = false;
        kparams.config = params.compile_config;
        kparams.device = device;
        kparams.mtl_func = mtl_func.get();
        kparams.mem_pool = params.mem_pool;
        kernel = std::make_unique<GcOpMtlKernel>(kparams);
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
            {ActionArg("ti_kernel_name", std::string(ti_kernel_attribs.name)),
             ActionArg("size_in_bytes", (int64)ctx_attribs.total_bytes())});
      }
      ctx_buffer =
          new_mtl_buffer_no_copy(device, ctx_mem->ptr(), ctx_mem->size());
    }
  }

  ~CompiledTaichiKernel() {
    for (auto [_, alloc_n_sz] : ext_arr_arg_to_dev_alloc) {
      rhi_device_->dealloc_memory(alloc_n_sz.alloc);
    }
  }

  void maybe_make_dev_alloc_for_ext_arrs(const RuntimeContext &host_ctx) {
    for (const auto &arg : ctx_attribs.args()) {
      if (!arg.is_array) {
        continue;
      }
      const int arg_id = arg.index;
      if (host_ctx.is_device_allocations[arg_id]) {
        continue;
      }
      // Even in the face that external array has 0-length, we still allocate
      // something, to prevent runtime edge cases.
      const auto arr_sz =
          std::max(host_ctx.array_runtime_sizes[arg.index], (uint64)4);
      auto itr = ext_arr_arg_to_dev_alloc.find(arg_id);
      const bool already_allocated = (itr != ext_arr_arg_to_dev_alloc.end());
      if (already_allocated && (itr->second.size >= arr_sz)) {
        continue;
      }
      if (already_allocated) {
        TI_TRACE("Dealloc dev_alloc for extarr size old={} new={}",
                 itr->second.size, arr_sz);
        rhi_device_->dealloc_memory(itr->second.alloc);
      }
      // This are the device buffers for "ext_arr". Unlike Ndarray, an ext_arr
      // is actually a template param, so its size is always fixed per
      // instantiated kernel.
      Device::AllocParams aparams;
      aparams.size = arr_sz;
      aparams.host_read = true;
      aparams.host_write = true;
      AllocAndSize alloc_n_sz;
      alloc_n_sz.alloc = rhi_device_->allocate_memory(aparams);
      alloc_n_sz.size = aparams.size;
      ext_arr_arg_to_dev_alloc[arg_id] = alloc_n_sz;
    }
  }

  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  std::vector<std::unique_ptr<CompiledMtlKernelBase>> compiled_mtl_kernels;
  TaichiKernelAttributes ti_kernel_attribs;
  KernelContextAttributes ctx_attribs;
  std::unique_ptr<BufferMemoryView> ctx_mem;
  nsobj_unique_ptr<MTLBuffer> ctx_buffer;

  struct AllocAndSize {
    DeviceAllocation alloc;
    size_t size{0};
  };
  std::unordered_map<int, AllocAndSize> ext_arr_arg_to_dev_alloc;

 private:
  const Kernel *const kernel_;
  Device *const rhi_device_;
};

class HostMetalCtxBlitter {
 public:
  HostMetalCtxBlitter(const CompiledTaichiKernel &kernel,
                      RuntimeContext *host_ctx,
                      Device *rhi_device,
                      AllocToMTLBufferMapper *alloc_mapper,
                      uint64_t *host_result_buffer,
                      const std::string &kernel_name)
      : cti_kernel_(&kernel),
        ti_kernel_attribs_(&kernel.ti_kernel_attribs),
        ctx_attribs_(&kernel.ctx_attribs),
        host_ctx_(host_ctx),
        rhi_device_(rhi_device),
        alloc_mapper_(alloc_mapper),
        host_result_buffer_(host_result_buffer),
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
    struct BufferAndSize {
      MTLBuffer *buffer{nullptr};
      size_t size{0};
    };
    std::vector<BufferAndSize> buf_sz;
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      const auto dt = arg.dt;
      char *device_ptr = base + arg.offset_in_mem;
      if (!ti_kernel_attribs_->is_jit_evaluator) {
        ActionRecorder::get_instance().record(
            "context_host_to_metal",
            {ActionArg("ti_kernel_name", kernel_name_), ActionArg("arg_id", i),
             ActionArg("offset_in_bytes", (int64)arg.offset_in_mem)});
      }
      if (arg.is_array) {
        if (host_ctx_->is_device_allocations[i]) {
          // There is no way to write from host into Ndarray directly (yet), so
          // we don't have to do anything here.
        } else {
          const void *host_ptr = host_ctx_->get_arg<void *>(i);
          const auto alloc_n_sz = cti_kernel_->ext_arr_arg_to_dev_alloc.at(i);
          auto buf_mem = alloc_mapper_->find(alloc_n_sz.alloc);
          TI_ASSERT(buf_mem.buffer != nullptr);
          auto *mem = buf_mem.mem;
          TI_ASSERT(mem != nullptr);
          // NOTE: DO NOT use `alloc_n_sz.size`. That is the size for the
          // allocation, NOT the array size.
          const auto arr_size = host_ctx_->array_runtime_sizes[i];
          std::memcpy(mem->ptr(), host_ptr, arr_size);
          buf_sz.push_back(BufferAndSize{buf_mem.buffer, arr_size});
        }
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
    buf_sz.push_back(
        BufferAndSize{kernel_ctx_buffer_, kernel_ctx_mem_->size()});
    for (auto bs : buf_sz) {
      did_modify_range(bs.buffer, /*length=*/0, bs.size);
    }
  }

  void metal_to_host() {
#define TO_HOST(type, offset)                                      \
  const type d = *(reinterpret_cast<type *>(device_ptr) + offset); \
  host_result_buffer_[offset] =                                    \
      taichi_union_cast_with_different_sizes<uint64>(d);

    if (ctx_attribs_->empty()) {
      return;
    }
    char *const base = (char *)kernel_ctx_mem_->ptr();
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      if (arg.is_array) {
        if (host_ctx_->is_device_allocations[i]) {
          continue;
        }
        const auto alloc_n_sz = cti_kernel_->ext_arr_arg_to_dev_alloc.at(i);
        auto *mem = alloc_mapper_->find(alloc_n_sz.alloc).mem;
        TI_ASSERT(mem != nullptr);
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        const auto arr_size = host_ctx_->array_runtime_sizes[i];
        std::memcpy(host_ptr, mem->ptr(), arr_size);
        if (!ti_kernel_attribs_->is_jit_evaluator) {
          ActionRecorder::get_instance().record(
              "context_metal_to_host",
              {
                  ActionArg("ti_kernel_name", kernel_name_),
                  ActionArg("arg_id", i),
                  ActionArg("arg_type", "ptr"),
                  ActionArg("size_in_bytes", (int64)arg.stride),
              });
        }
      }
    }
    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = base + ret.offset_in_mem;
      const int dt_bytes = metal_data_type_bytes(ret.dt);
      const int num = ret.stride / dt_bytes;
      for (int j = 0; j < num; ++j) {
        const auto dt = ret.dt;
        if (dt == MetalDataType::i32) {
          TO_HOST(int32, j);
        } else if (dt == MetalDataType::u32) {
          TO_HOST(uint32, j);
        } else if (dt == MetalDataType::f32) {
          TO_HOST(float32, j);
        } else if (dt == MetalDataType::i8) {
          TO_HOST(int8, j);
        } else if (dt == MetalDataType::i16) {
          TO_HOST(int16, j);
        } else if (dt == MetalDataType::u8) {
          TO_HOST(uint8, j);
        } else if (dt == MetalDataType::u16) {
          TO_HOST(uint16, j);
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
      RuntimeContext *ctx,
      Device *rhi_device,
      AllocToMTLBufferMapper *alloc_mapper,
      uint64_t *host_result_buffer,
      std::string name) {
    if (kernel.ctx_attribs.empty()) {
      return nullptr;
    }
    return std::make_unique<HostMetalCtxBlitter>(
        kernel, ctx, rhi_device, alloc_mapper, host_result_buffer, name);
  }

 private:
  const CompiledTaichiKernel *const cti_kernel_;
  const TaichiKernelAttributes *const ti_kernel_attribs_;
  const KernelContextAttributes *const ctx_attribs_;
  RuntimeContext *const host_ctx_;
  Device *const rhi_device_;
  AllocToMTLBufferMapper *const alloc_mapper_;
  uint64_t *const host_result_buffer_;
  BufferMemoryView *const kernel_ctx_mem_;
  MTLBuffer *const kernel_ctx_buffer_;
  std::string kernel_name_;
};

struct DevAllocWithInternals {
  DeviceAllocation handle;
  BufferMemoryView *mem{nullptr};
  MTLBuffer *buffer{nullptr};
};

}  // namespace

class KernelManager::Impl {
 public:
  explicit Impl(Params params)
      : config_(params.config),
        compiled_runtime_module_(params.compiled_runtime_module),
        mem_pool_(params.mem_pool),
        host_result_buffer_(params.host_result_buffer),
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

    {
      ComputeDeviceParams rhi_params;
      rhi_params.device = device_.get();
      rhi_params.mem_pool = mem_pool_;
      rhi_params.only_for_dev_allocation = true;
      auto make_res = make_compute_device(rhi_params);
      rhi_device_ = std::move(make_res.device);
      TI_ASSERT(rhi_device_ != nullptr);
      devalloc_mapper_ = make_res.mapper;
    }

    global_tmps_idevalloc_ = make_idevalloc(taichi_global_tmp_buffer_size);

    ActionRecorder::get_instance().record(
        "allocate_global_tmp_buffer",
        {ActionArg("size_in_bytes", (int64)taichi_global_tmp_buffer_size)});

    const size_t mem_pool_bytes =
        (config_->device_memory_GB * 1024 * 1024 * 1024ULL);
    runtime_idevalloc_ =
        make_idevalloc(compiled_runtime_module_.runtime_size + mem_pool_bytes);
    buffer_meta_data_.runtime_buffer_size =
        compiled_runtime_module_.runtime_size;
    TI_DEBUG(
        "Metal runtime buffer size: {} bytes (sizeof(Runtime)={} "
        "memory_pool={})",
        runtime_idevalloc_.mem->size(), compiled_runtime_module_.runtime_size,
        mem_pool_bytes);

    ActionRecorder::get_instance().record(
        "allocate_runtime_buffer",
        {ActionArg("runtime_buffer_size_in_bytes",
                   (int64)runtime_idevalloc_.mem->size()),
         ActionArg("runtime_size_in_bytes",
                   (int64)compiled_runtime_module_.runtime_size),
         ActionArg("memory_pool_size", (int64)mem_pool_bytes)});

    TI_ASSERT_INFO(runtime_idevalloc_.buffer != nullptr,
                   "Failed to allocate Metal runtime buffer");
    print_assert_idevalloc_ =
        make_idevalloc(shaders::kMetalPrintAssertBufferSize);
    TI_ASSERT(print_assert_idevalloc_.buffer != nullptr);

    init_runtime_buffer(compiled_runtime_module_, params.config->random_seed);
    clear_print_assert_buffer();
  }

  ~Impl() {
    for (auto &rb : root_buffers_) {
      rhi_device_->dealloc_memory(rb.idevalloc.handle);
    }
    rhi_device_->dealloc_memory(global_tmps_idevalloc_.handle);
    rhi_device_->dealloc_memory(runtime_idevalloc_.handle);
    rhi_device_->dealloc_memory(print_assert_idevalloc_.handle);
  }

  void add_compiled_snode_tree(const CompiledStructs &compiled_tree) {
    SNodesRootBuffer rtbuf{};
    rtbuf.desc = BufferDescriptor::root(compiled_tree.root_id);
    if (compiled_tree.root_size > 0) {
      rtbuf.idevalloc = make_idevalloc(compiled_tree.root_size);
      const auto buf_sz = rtbuf.idevalloc.mem->size();
      buffer_meta_data_.root_buffer_size += buf_sz;
      TI_DEBUG("Metal root={} buffer_size={} bytes", compiled_tree.root_id,
               buf_sz);
      ActionRecorder::get_instance().record(
          "allocate_root_buffer",
          {ActionArg("root_id={}", (int64)compiled_tree.root_id),
           ActionArg("size_in_bytes", (int64)buf_sz)});
    }
    init_snode_tree_sparse_runtime(compiled_tree);

    compiled_snode_trees_.push_back(compiled_tree);
    root_buffers_.push_back(std::move(rtbuf));
  }

  void register_taichi_kernel(const std::string &taichi_kernel_name,
                              const std::string &mtl_kernel_source_code,
                              const TaichiKernelAttributes &ti_kernel_attribs,
                              const KernelContextAttributes &ctx_attribs,
                              const Kernel *kernel) {
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
    params.device = device_.get();
    params.mem_pool = mem_pool_;
    params.profiler = profiler_;
    params.compile_config = config_;
    params.kernel = kernel;
    params.rhi_device = rhi_device_.get();
    compiled_taichi_kernels_[taichi_kernel_name] =
        std::make_unique<CompiledTaichiKernel>(params);
    TI_DEBUG("Registered Taichi kernel <{}>", taichi_kernel_name);
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            RuntimeContext *ctx) {
    mac::ScopedAutoreleasePool pool;
    auto &cti_kernel =
        *compiled_taichi_kernels_.find(taichi_kernel_name)->second;
    cti_kernel.maybe_make_dev_alloc_for_ext_arrs(*ctx);
    auto ctx_blitter = HostMetalCtxBlitter::maybe_make(
        cti_kernel, ctx, rhi_device_.get(), devalloc_mapper_,
        host_result_buffer_, taichi_kernel_name);
    if (config_->verbose_kernel_launches) {
      TI_INFO("Launching Taichi kernel <{}>", taichi_kernel_name);
    }

    InputBuffersMap input_buffers;
    for (auto &rb : root_buffers_) {
      input_buffers[rb.desc] = rb.idevalloc.buffer;
    }
    input_buffers[BufferDescriptor::global_tmps()] =
        global_tmps_idevalloc_.buffer;
    input_buffers[BufferDescriptor::runtime()] = runtime_idevalloc_.buffer;
    input_buffers[BufferDescriptor::print()] = print_assert_idevalloc_.buffer;

    if (ctx_blitter) {
      ctx_blitter->host_to_metal();
      input_buffers[BufferDescriptor::context()] = cti_kernel.ctx_buffer.get();
    }
    auto ndarray_buffers = get_dev_alloc_buffers(cti_kernel, *ctx);
    input_buffers.insert(ndarray_buffers.begin(), ndarray_buffers.end());

    for (const auto &mk : cti_kernel.compiled_mtl_kernels) {
      mk->launch(input_buffers, cur_command_buffer_.get());
    }

    const auto &used = cti_kernel.ti_kernel_attribs.used_features;
    const bool used_print_assert = (used.print || used.assertion);
    if (ctx_blitter || used_print_assert) {
      // TODO(k-ye): One optimization is to synchronize only when we absolutely
      // need to transfer the data back to host. This includes the cases where
      // an arg is 1) an array, or 2) used as return value.
      std::vector<MTLBuffer *> buffers_to_blit;
      if (ctx_blitter) {
        buffers_to_blit.push_back(ctx_blitter->ctx_buffer());
      }
      if (used_print_assert) {
        clear_print_assert_buffer();
        buffers_to_blit.push_back(print_assert_idevalloc_.buffer);
      }
      for (auto [_, buf] : ndarray_buffers) {
        buffers_to_blit.push_back(buf);
      }
      blit_buffers_and_sync(buffers_to_blit);

      if (ctx_blitter) {
        ctx_blitter->metal_to_host();
      }
      if (used.assertion) {
        check_assertion_failure();
      }
      if (used.print) {
        flush_print_buffers();
      }
    }
  }

  void synchronize() {
    mac::ScopedAutoreleasePool pool;
    blit_buffers_and_sync();
  }

  BufferMetaData get_buffer_meta_data() {
    return buffer_meta_data_;
  }

  PrintStringTable *print_strtable() {
    return &print_strtable_;
  }

  std::size_t get_snode_num_dynamically_allocated(SNode *snode) {
    // TODO(k-ye): Have a generic way for querying these sparse runtime stats.
    mac::ScopedAutoreleasePool pool;
    blit_buffers_and_sync({runtime_idevalloc_.buffer});
    auto *sna = dev_runtime_mirror_.snode_allocators + snode->id;
    // WHY -1?
    //
    // We allocate one ambient element for each `pointer` SNode from its
    // corresponding snode_allocator |sna|. Therefore the count starts at 1.
    return sna->data_list.next - 1;
  }

  DeviceAllocation allocate_memory(const Device::AllocParams &params) {
    auto res = rhi_device_->allocate_memory(params);
    return res;
  }

 private:
  void init_runtime_buffer(const CompiledRuntimeModule &rtm_module,
                           int random_seed) {
    char *addr = runtime_idevalloc_.mem->ptr();
    // init rand_seeds
    std::default_random_engine generator((unsigned int)random_seed);
    std::uniform_int_distribution<uint32_t> distr(
        0, std::numeric_limits<uint32_t>::max());
    for (int i = 0; i < kNumRandSeeds; ++i) {
      uint32_t *s = reinterpret_cast<uint32_t *>(addr);
      *s = distr(generator);
      addr += sizeof(uint32_t);
    }
    TI_DEBUG("Initialized random seeds size={}", rtm_module.rand_seeds_size);

    using namespace shaders;
    addr = runtime_idevalloc_.mem->ptr() + rtm_module.rand_seeds_size;
    const char *const addr_begin = addr;
    dev_runtime_mirror_.snode_metas = (SNodeMeta *)addr;
    size_t addr_offset = sizeof(SNodeMeta) * kMaxNumSNodes;
    addr += addr_offset;
    TI_DEBUG("SNodeMeta, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    dev_runtime_mirror_.snode_extractors = (SNodeExtractors *)addr;
    addr_offset = sizeof(SNodeExtractors) * kMaxNumSNodes;
    addr += addr_offset;
    TI_DEBUG("SNodeExtractors, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    dev_runtime_mirror_.snode_lists = (ListManagerData *)addr;
    addr_offset = sizeof(ListManagerData) * kMaxNumSNodes;
    addr += addr_offset;
    TI_DEBUG("ListManagerData, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    dev_runtime_mirror_.snode_allocators = (NodeManagerData *)addr;
    addr_offset = sizeof(NodeManagerData) * kMaxNumSNodes;
    addr += addr_offset;
    TI_DEBUG("NodeManagerData, size={} accumulated={}", addr_offset,
             (addr - addr_begin));
    dev_runtime_mirror_.ambient_indices = (NodeManagerData::ElemIndex *)addr;
    addr_offset = sizeof(NodeManagerData::ElemIndex) * kMaxNumSNodes;
    addr += addr_offset;
    TI_DEBUG("SNode ambient elements, size={} accumulated={}", addr_offset,
             (addr - addr_begin));

    // Initialize the memory allocator
    dev_mem_alloc_mirror_ = reinterpret_cast<MemoryAllocator *>(addr);
    // Make sure the retured memory address is always greater than 1.
    dev_mem_alloc_mirror_->next = shaders::MemoryAllocator::kInitOffset;
    TI_DEBUG("Memory allocator, begin={} next={}", (addr - addr_begin),
             dev_mem_alloc_mirror_->next);

    mark_runtime_buffer_modified();
  }

  void init_snode_tree_sparse_runtime(const CompiledStructs &snode_tree) {
    using namespace shaders;
    const int max_snodes = snode_tree.max_snodes;
    const auto &snode_descriptors = snode_tree.snode_descriptors;
    char *addr = nullptr;
    // init snode_metas
    addr = (char *)dev_runtime_mirror_.snode_metas;
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
        case SNodeType::pointer:
          rtm_meta->type = SNodeMeta::Pointer;
          break;
        case SNodeType::bit_struct:
          rtm_meta->type = SNodeMeta::BitStruct;
          break;
        default:
          TI_ERROR("Unsupported SNode type={}",
                   snode_type_name(sn_meta.snode->type));
          break;
      }
      TI_DEBUG(
          "SnodeMeta\n  id={}\n  type={}\n  element_stride={}\n  "
          "num_slots={}\n  mem_offset_in_parent={}\n",
          i, snode_type_name(sn_meta.snode->type), rtm_meta->element_stride,
          rtm_meta->num_slots, rtm_meta->mem_offset_in_parent);
    }
    // init snode_extractors
    addr = (char *)dev_runtime_mirror_.snode_extractors;
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
        rtm_ext->extractors[j].num_bits = ext.num_bits;
        rtm_ext->extractors[j].acc_offset = ext.acc_offset;
        rtm_ext->extractors[j].num_elements_from_root =
            ext.num_elements_from_root;
        TI_DEBUG("  [{}] num_bits={} acc_offset={} num_elements_from_root={}",
                 j, ext.num_bits, ext.acc_offset, ext.num_elements_from_root);
      }
      TI_DEBUG("");
    }
    // init snode_lists
    addr = (char *)dev_runtime_mirror_.snode_lists;
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
    // init snode_allocators
    addr = (char *)dev_runtime_mirror_.snode_allocators;
    auto init_node_mgr = [&snode_descriptors](const SNodeDescriptor &sn_desc,
                                              NodeManagerData *nm_data) {
      nm_data->data_list.element_stride = sn_desc.element_stride;
      const int num_elems_per_chunk = compute_num_elems_per_chunk(
          sn_desc.total_num_self_from_root(snode_descriptors));
      const int log2num = log2int(num_elems_per_chunk);
      nm_data->data_list.log2_num_elems_per_chunk = log2num;
      nm_data->data_list.next = 0;

      nm_data->free_list.element_stride = sizeof(int32_t);
      nm_data->free_list.log2_num_elems_per_chunk = log2num;
      nm_data->free_list.next = 0;

      nm_data->recycled_list.element_stride = sizeof(int32_t);
      nm_data->recycled_list.log2_num_elems_per_chunk = log2num;
      nm_data->recycled_list.next = 0;

      nm_data->recycled_list_size_backup = 0;
      TI_DEBUG(
          "NodeManagerData\n  id={}\n  element_stride={}\n  "
          "num_elems_per_chunk={}\n  log2num={}\n",
          sn_desc.snode->id, nm_data->data_list.element_stride,
          num_elems_per_chunk, log2num);
    };
    std::vector<std::pair<int, NodeManagerData *>> snode_id_to_nodemgrs;
    for (int i = 0; i < max_snodes; ++i) {
      auto iter = snode_descriptors.find(i);
      if (iter == snode_descriptors.end()) {
        continue;
      }
      const SNodeDescriptor &sn_desc = iter->second;
      if (sn_desc.snode->type == SNodeType::root) {
        // We have to skip root SNode for two reasons:
        // 1. Root SNode is never sparse, so it never uses NodeManager
        // 2. If not skippped, we would allocate one ambient element for the
        // root. Because the allocation is done by chunk, when the root is
        // large, this would result in memory pool overflow.
        continue;
      }
      NodeManagerData *nm_data = reinterpret_cast<NodeManagerData *>(addr) + i;
      init_node_mgr(sn_desc, nm_data);
      snode_id_to_nodemgrs.push_back(std::make_pair(i, nm_data));
    }

    // Root list is static, so it can be initialized here once.
    ListgenElement root_elem;
    root_elem.mem_offset = 0;
    for (int i = 0; i < taichi_max_num_indices; ++i) {
      root_elem.coords.at[i] = 0;
    }
    ListManager root_lm;
    root_lm.lm_data = rtm_list_begin + snode_tree.root_id;
    root_lm.mem_alloc = dev_mem_alloc_mirror_;
    root_lm.append(root_elem);

    // Initialize all the ambient elements
    auto *const ambient_indices_begin = dev_runtime_mirror_.ambient_indices;
    for (const auto &p : snode_id_to_nodemgrs) {
      NodeManager nm;
      nm.nm_data = p.second;
      nm.mem_alloc = dev_mem_alloc_mirror_;
      const auto snode_id = p.first;
      ambient_indices_begin[snode_id] = nm.allocate();
      TI_DEBUG("AmbientIndex\n  id={}\n  mem_alloc->next={}\n", snode_id,
               dev_mem_alloc_mirror_->next);
    }
    mark_runtime_buffer_modified();
  }

  void mark_runtime_buffer_modified() {
    did_modify_range(runtime_idevalloc_.buffer, /*location=*/0,
                     runtime_idevalloc_.mem->size());
  }

  void clear_print_assert_buffer() {
    const auto sz = print_assert_idevalloc_.mem->size();
    std::memset(print_assert_idevalloc_.mem->ptr(), 0, sz);
    did_modify_range(print_assert_idevalloc_.buffer, /*location=*/0, sz);
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
    if (profiler_)
      profiler_->start("metal_synchronize");
    commit_command_buffer(cur_command_buffer_.get());
    wait_until_completed(cur_command_buffer_.get());
    create_new_command_buffer();
    if (profiler_)
      profiler_->stop();

    // print_runtime_debug();
  }

  void check_assertion_failure() {
    // TODO: Copy this to program's result_buffer, and let the Taichi runtime
    // handle the assertion failures uniformly.
    auto *asst_rec = reinterpret_cast<shaders::AssertRecorderData *>(
        print_assert_idevalloc_.mem->ptr());
    if (!asst_rec->flag) {
      return;
    }
    auto *msg_ptr = reinterpret_cast<int32_t *>(asst_rec + 1);
    shaders::PrintMsg msg(msg_ptr, asst_rec->num_args);
    using MsgType = shaders::PrintMsg::Type;
    TI_ASSERT(msg.pm_get_type(0) == MsgType::Str);
    const auto fmt_str = print_strtable_.get(msg.pm_get_data(0));
    const auto err_str = format_error_message(fmt_str, [&msg](int argument_id) {
      // +1 to skip the first arg, which is the error message template.
      const int32 x = msg.pm_get_data(argument_id + 1);
      return taichi_union_cast_with_different_sizes<uint64>(x);
    });
    // Note that we intentionally comment out the flag reset below, because it
    // is ineffective at all. This is a very tricky part:
    // 1. Under .managed storage mode, we need to call [didModifyRange:] to sync
    // buffer data from CPU -> GPU. So ideally, after resetting the flag, we
    // should just do so.
    // 2. However, during the assertion (TI_ERROR), the stack unwinding seems to
    // have deviated from the normal execution path. As a result, if we put
    // [didModifyRange:] after TI_ERROR, it doesn't get executed...
    // 3. The reason we put [didModifyRange:] after TI_ERROR is because we
    // should do so after flush_print_buffers():
    //
    //   check_assertion_failure();  <-- Code below is skipped...
    //   flush_print_buffers();
    //   memset(print_assert_idevalloc_.mem->ptr(), 0,
    //          print_assert_idevalloc_.mem->size());
    //   did_modify_range(print_assert_idevalloc_.buffer);
    //
    // As a workaround, we put [didModifyRange:] before sync, where the program
    // is still executing normally.
    // asst_rec->flag = 0;
    throw TaichiAssertionError(err_str);
  }

  void flush_print_buffers() {
    auto *pa = reinterpret_cast<shaders::PrintMsgAllocator *>(
        print_assert_idevalloc_.mem->ptr() + shaders::kMetalAssertBufferSize);
    const int used_sz =
        std::min(pa->next, shaders::kMetalPrintMsgsMaxQueueSize);
    TI_TRACE("Print buffer used bytes: {}", used_sz);
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
        } else if (dt == MsgType::U32) {
          py_cout << static_cast<uint32_t>(x);
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

    if (pa->next >= shaders::kMetalPrintMsgsMaxQueueSize) {
      py_cout << "...(maximum print buffer reached)\n";
    }

    // Comment out intentionally since it is ineffective otherwise. See
    // check_assertion_failure() for the explanation.
    // pa->next = 0;
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
    return *reinterpret_cast<const T *>(
        (const char *)global_tmps_idevalloc_.mem->ptr() + offset);
  }

  DevAllocWithInternals make_idevalloc(size_t size) {
    Device::AllocParams params;
    // host_read|write honestly don't matter at this point, because the Metal
    // backend only uses .managed mode.
    params.host_read = false;
    params.host_write = false;
    params.size = size;
    params.usage = AllocUsage::Storage;
    DevAllocWithInternals res;
    res.handle = rhi_device_->allocate_memory(params);
    auto bm = devalloc_mapper_->find(res.handle);
    res.buffer = bm.buffer;
    res.mem = bm.mem;
    return res;
  }

  InputBuffersMap get_dev_alloc_buffers(const CompiledTaichiKernel &ctk,
                                        const RuntimeContext &host_ctx) const {
    InputBuffersMap res;
    for (const auto &arg : ctk.ctx_attribs.args()) {
      if (!arg.is_array) {
        continue;
      }
      DeviceAllocation dev_alloc;
      if (host_ctx.is_device_allocations[arg.index]) {
        dev_alloc = *reinterpret_cast<const DeviceAllocation *>(
            host_ctx.args[arg.index]);
        TI_TRACE("Ndarray arg_id={} alloc_id={}", arg.index,
                 dev_alloc.alloc_id);
      } else {
        dev_alloc = ctk.ext_arr_arg_to_dev_alloc.at(arg.index).alloc;
        TI_TRACE("ExtArr arg_id={} alloc_id={}", arg.index, dev_alloc.alloc_id);
      }
      MTLBuffer *buffer = devalloc_mapper_->find(dev_alloc).buffer;
      TI_ASSERT(buffer != nullptr);
      res[BufferDescriptor::ndarray(arg.index)] = buffer;
    }
    return res;
  }

  struct SNodesRootBuffer {
    BufferDescriptor desc;
    DevAllocWithInternals idevalloc;
  };

  CompileConfig *const config_;
  const CompiledRuntimeModule compiled_runtime_module_;
  MemoryPool *const mem_pool_;
  uint64_t *const host_result_buffer_;
  KernelProfilerBase *const profiler_;
  // TODO(k-ye): Name this to `device_`, then hide the `MTLDevice device_`.
  std::unique_ptr<Device> rhi_device_{nullptr};
  AllocToMTLBufferMapper *devalloc_mapper_{nullptr};
  // FIXME: This is for AOT, name it better
  BufferMetaData buffer_meta_data_;
  std::vector<CompiledStructs> compiled_snode_trees_;
  nsobj_unique_ptr<MTLDevice> device_{nullptr};
  nsobj_unique_ptr<MTLCommandQueue> command_queue_{nullptr};
  nsobj_unique_ptr<MTLCommandBuffer> cur_command_buffer_{nullptr};
  std::size_t command_buffer_id_{0};
  std::vector<SNodesRootBuffer> root_buffers_;
  DevAllocWithInternals global_tmps_idevalloc_;
  DevAllocWithInternals runtime_idevalloc_;
  DevAllocWithInternals print_assert_idevalloc_;
  int last_snode_id_used_in_runtime_{-1};
  std::unordered_map<std::string, std::unique_ptr<CompiledTaichiKernel>>
      compiled_taichi_kernels_;
  PrintStringTable print_strtable_;

  // The |dev_*_mirror_|s are the data structures stored in the Metal device
  // side that get mirrored to the host side. This is possible because the
  // underlying memory between host and device is unified. However, make sure
  // to do a dev <-> host buffer synchronization before reading from/after
  // writing to these mirrors.
  //
  // TODO(k-ye): These mirrors are really just a few pointers into the memory
  // region maintained by |runtime_mem_|. Maybe create a view wrapper directly
  // on top of |runtime_mem_|?
  shaders::Runtime dev_runtime_mirror_;
  shaders::MemoryAllocator *dev_mem_alloc_mirror_{nullptr};
};

#else

class KernelManager::Impl {
 public:
  explicit Impl(Params) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void add_compiled_snode_tree(const CompiledStructs &) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void register_taichi_kernel(const std::string &taichi_kernel_name,
                              const std::string &mtl_kernel_source_code,
                              const TaichiKernelAttributes &ti_kernel_attribs,
                              const KernelContextAttributes &ctx_attribs,
                              const Kernel *kernel) {
    TI_ERROR("Metal not supported on the current OS");
  }

  BufferMetaData get_buffer_meta_data() {
    TI_ERROR("Metal not supported on the current OS");
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            RuntimeContext *ctx) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void synchronize() {
    TI_ERROR("Metal not supported on the current OS");
  }

  PrintStringTable *print_strtable() {
    TI_ERROR("Metal not supported on the current OS");
    return nullptr;
  }

  std::size_t get_snode_num_dynamically_allocated(SNode *) {
    TI_ERROR("Metal not supported on the current OS");
    return 0;
  }

  DeviceAllocation allocate_memory(const Device::AllocParams &params) {
    TI_ERROR("Metal not supported on the current OS");
    return kDeviceNullAllocation;
  }
};

#endif  // TI_PLATFORM_OSX

KernelManager::KernelManager(Params params)
    : impl_(std::make_unique<Impl>(std::move(params))) {
}

KernelManager::~KernelManager() {
}

void KernelManager::add_compiled_snode_tree(const CompiledStructs &snode_tree) {
  impl_->add_compiled_snode_tree(snode_tree);
}

void KernelManager::register_taichi_kernel(
    const std::string &taichi_kernel_name,
    const std::string &mtl_kernel_source_code,
    const TaichiKernelAttributes &ti_kernel_attribs,
    const KernelContextAttributes &ctx_attribs,
    const Kernel *kernel) {
  impl_->register_taichi_kernel(taichi_kernel_name, mtl_kernel_source_code,
                                ti_kernel_attribs, ctx_attribs, kernel);
}

void KernelManager::launch_taichi_kernel(const std::string &taichi_kernel_name,
                                         RuntimeContext *ctx) {
  impl_->launch_taichi_kernel(taichi_kernel_name, ctx);
}

void KernelManager::synchronize() {
  impl_->synchronize();
}

BufferMetaData KernelManager::get_buffer_meta_data() {
  return impl_->get_buffer_meta_data();
}

PrintStringTable *KernelManager::print_strtable() {
  return impl_->print_strtable();
}

std::size_t KernelManager::get_snode_num_dynamically_allocated(SNode *snode) {
  return impl_->get_snode_num_dynamically_allocated(snode);
}

DeviceAllocation KernelManager::allocate_memory(
    const Device::AllocParams &params) {
  return impl_->allocate_memory(params);
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
