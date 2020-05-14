#include "taichi/backends/metal/kernel_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>

#include "taichi/backends/metal/constants.h"
#include "taichi/inc/constants.h"
#include "taichi/math/arithmetic.h"

#ifdef TI_PLATFORM_OSX
#include <sys/mman.h>
#include <unistd.h>

#include "taichi/backends/metal/api.h"
#endif  // TI_PLATFORM_OSX

TLANG_NAMESPACE_BEGIN
namespace metal {

#ifdef TI_PLATFORM_OSX

namespace {
namespace shaders {
#include "taichi/backends/metal/shaders/runtime_utils.metal.h"
}

using KernelTaskType = OffloadedStmt::TaskType;
using BufferEnum = KernelAttributes::Buffers;

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
    const KernelAttributes *kerenl_attribs;
    MTLDevice *device;
    MTLFunction *mtl_func;
  };

  explicit CompiledMtlKernelBase(Params &params)
      : kernel_attribs_(*params.kerenl_attribs),
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
    const int num_threads_per_group =
        get_max_total_threads_per_threadgroup(pipeline_state_.get());
    const int num_groups =
        ((num_threads + num_threads_per_group - 1) / num_threads_per_group);
    dispatch_threadgroups(encoder.get(), num_groups,
                          std::min(num_threads, num_threads_per_group));
    end_encoding(encoder.get());
  }

  KernelAttributes kernel_attribs_;
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
    const SNode *snode = nullptr;
  };

  explicit RuntimeListOpsMtlKernel(Params &params)
      : CompiledMtlKernelBase(params),
        parent_snode_id_(params.snode->parent->id),
        child_snode_id_(params.snode->id),
        args_mem_(std::make_unique<BufferMemoryView>(
            /*size=*/sizeof(int32_t) * 2,
            params.mem_pool)),
        args_buffer_(new_mtl_buffer_no_copy(params.device,
                                            args_mem_->ptr(),
                                            args_mem_->size())) {
    TI_ASSERT(args_buffer_ != nullptr);
    auto *mem = reinterpret_cast<int32_t *>(args_mem_->ptr());
    mem[0] = parent_snode_id_;
    mem[1] = child_snode_id_;
  }

  void launch(InputBuffersMap &input_buffers,
              MTLCommandBuffer *command_buffer) override {
    BindBuffers buffers;
    for (const auto b : kernel_attribs_.buffers) {
      if (b == BufferEnum::Args) {
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
    std::string_view taichi_kernel_name;
    std::string mtl_source_code;
    const std::vector<KernelAttributes> *mtl_kernels_attribs;
    const KernelContextAttributes *args_attribs;
    MTLDevice *device;
    MemoryPool *mem_pool;
    ProfilerBase *profiler;
  };

  CompiledTaichiKernel(Params params) : args_attribs(*params.args_attribs) {
    auto *const device = params.device;
    auto kernel_lib = new_library_with_source(device, params.mtl_source_code);
    if (kernel_lib == nullptr) {
      TI_ERROR("Failed to compile Metal kernel! Generated code:\n\n{}",
               params.mtl_source_code);
    }
    for (const auto &ka : *(params.mtl_kernels_attribs)) {
      auto mtl_func = new_function_with_name(kernel_lib.get(), ka.name);
      TI_ASSERT(mtl_func != nullptr);
      // Note that CompiledMtlKernel doesn't own |kernel_func|.
      std::unique_ptr<CompiledMtlKernelBase> kernel = nullptr;
      const auto ktype = ka.task_type;
      if (ktype == KernelTaskType::clear_list ||
          ktype == KernelTaskType::listgen) {
        RuntimeListOpsMtlKernel::Params kparams;
        kparams.kerenl_attribs = &ka;
        kparams.device = device;
        kparams.mtl_func = mtl_func.get();
        kparams.snode = ka.runtime_list_op_attribs.snode;
        kparams.mem_pool = params.mem_pool;
        kernel = std::make_unique<RuntimeListOpsMtlKernel>(kparams);
      } else {
        CompiledMtlKernelBase::Params kparams;
        kparams.kerenl_attribs = &ka;
        kparams.device = device;
        kparams.mtl_func = mtl_func.get();
        kernel = std::make_unique<UserMtlKernel>(kparams);
      }

      TI_ASSERT(kernel != nullptr);
      compiled_mtl_kernels.push_back(std::move(kernel));
      TI_DEBUG("Added {} for Taichi kernel {}", ka.debug_string(),
               params.taichi_kernel_name);
    }
    if (!args_attribs.empty()) {
      args_mem = std::make_unique<BufferMemoryView>(args_attribs.total_bytes(),
                                                    params.mem_pool);
      args_buffer =
          new_mtl_buffer_no_copy(device, args_mem->ptr(), args_mem->size());
    }
  }

  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  std::vector<std::unique_ptr<CompiledMtlKernelBase>> compiled_mtl_kernels;
  KernelContextAttributes args_attribs;
  std::unique_ptr<BufferMemoryView> args_mem;
  nsobj_unique_ptr<MTLBuffer> args_buffer;

 private:
};

class HostMetalCtxBlitter {
 public:
  HostMetalCtxBlitter(const KernelContextAttributes *args_attribs,
                      Context *ctx,
                      BufferMemoryView *args_mem)
      : ctx_attribs_(args_attribs), ctx_(ctx), args_mem_(args_mem) {
  }

  void host_to_metal() {
#define TO_METAL(type)             \
  auto d = ctx_->get_arg<type>(i); \
  std::memcpy(device_ptr, &d, sizeof(d))

    if (ctx_attribs_->empty()) {
      return;
    }
    char *const base = (char *)args_mem_->ptr();
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
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
    char *device_ptr = base + ctx_attribs_->ctx_bytes();
    std::memcpy(device_ptr, ctx_->extra_args, ctx_attribs_->extra_args_bytes());
#undef TO_METAL
  }

  void metal_to_host() {
#define TO_HOST(type)                                   \
  const type d = *reinterpret_cast<type *>(device_ptr); \
  ctx_->set_arg<type>(i, d)

    if (ctx_attribs_->empty()) {
      return;
    }
    char *const base = (char *)args_mem_->ptr();
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      char *device_ptr = base + arg.offset_in_mem;
      if (arg.is_array) {
        void *host_ptr = ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);
      }
    }
    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = base + ret.offset_in_mem;
      if (ret.is_array) {
        void *host_ptr = ctx_->get_arg<void *>(i);
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
      Context *ctx) {
    if (kernel.args_attribs.empty()) {
      return nullptr;
    }
    return std::make_unique<HostMetalCtxBlitter>(&kernel.args_attribs, ctx,
                                                 kernel.args_mem.get());
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  Context *const ctx_;
  BufferMemoryView *const args_mem_;
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
    }

    global_tmps_mem_ = std::make_unique<BufferMemoryView>(
        taichi_global_tmp_buffer_size, mem_pool_);
    global_tmps_buffer_ = new_mtl_buffer_no_copy(
        device_.get(), global_tmps_mem_->ptr(), global_tmps_mem_->size());
    TI_ASSERT(global_tmps_buffer_ != nullptr);

    if (compiled_structs_.runtime_size > 0) {
      runtime_mem_ = std::make_unique<BufferMemoryView>(
          compiled_structs_.runtime_size, mem_pool_);
      runtime_buffer_ = new_mtl_buffer_no_copy(
          device_.get(), runtime_mem_->ptr(), runtime_mem_->size());
      TI_DEBUG("Metal runtime buffer size: {} bytes", runtime_mem_->size());
      TI_ASSERT_INFO(
          runtime_buffer_ != nullptr,
          "Failed to allocate Metal runtime buffer, requested {} bytes",
          runtime_mem_->size());
      init_runtime(params.root_id);
    }
  }

  void register_taichi_kernel(
      const std::string &taichi_kernel_name,
      const std::string &mtl_kernel_source_code,
      const std::vector<KernelAttributes> &kernels_attribs,
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
    params.taichi_kernel_name = taichi_kernel_name;
    params.mtl_source_code = mtl_kernel_source_code;
    params.mtl_kernels_attribs = &kernels_attribs;
    params.args_attribs = &ctx_attribs;
    params.device = device_.get();
    params.mem_pool = mem_pool_;
    params.profiler = profiler_;
    compiled_taichi_kernels_[taichi_kernel_name] =
        std::make_unique<CompiledTaichiKernel>(params);
    TI_DEBUG("Registered Taichi kernel <{}>", taichi_kernel_name);
  }

  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            Context *ctx) {
    auto &ctk = *compiled_taichi_kernels_.find(taichi_kernel_name)->second;
    auto ctx_blitter = HostMetalCtxBlitter::maybe_make(ctk, ctx);
    if (config_->verbose_kernel_launches) {
      TI_INFO("Lauching Taichi kernel <{}>", taichi_kernel_name);
    }

    InputBuffersMap input_buffers = {
        {BufferEnum::Root, root_buffer_.get()},
        {BufferEnum::GlobalTmps, global_tmps_buffer_.get()},
        {BufferEnum::Runtime, runtime_buffer_.get()},
    };
    if (ctx_blitter) {
      ctx_blitter->host_to_metal();
      input_buffers[BufferEnum::Args] = ctk.args_buffer.get();
    }
    for (const auto &mk : ctk.compiled_mtl_kernels) {
      mk->launch(input_buffers, cur_command_buffer_.get());
    }
    if (ctx_blitter) {
      // TODO(k-ye): One optimization is to synchronize only when we absolutely
      // need to transfer the data back to host. This includes the cases where
      // an arg is 1) an array, or 2) used as return value.
      synchronize();
      ctx_blitter->metal_to_host();
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
  void init_runtime(int root_id) {
    using namespace shaders;
    char *addr = reinterpret_cast<char *>(runtime_mem_->ptr());
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
      TI_DEBUG("SnodeMeta\n  id={}\n  element_stride={}\n  num_slots={}\n", i,
               rtm_meta->element_stride, rtm_meta->num_slots);
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
        default:
          TI_ERROR("Unsupported SNode type={}",
                   snode_type_name(sn_meta.snode->type));
          break;
      }
    }
    addr += sizeof(SNodeMeta) * max_snodes;
    // init snode_extractors
    for (int i = 0; i < max_snodes; ++i) {
      auto iter = snode_descriptors.find(i);
      if (iter == snode_descriptors.end()) {
        continue;
      }
      const auto *sn = iter->second.snode;
      SNodeExtractors *rtm_ext = reinterpret_cast<SNodeExtractors *>(addr) + i;
      for (int j = 0; j < taichi_max_num_indices; ++j) {
        const auto &ext = sn->extractors[j];
        rtm_ext->extractors[j].start = ext.start;
        rtm_ext->extractors[j].num_bits = ext.num_bits;
        rtm_ext->extractors[j].acc_offset = ext.acc_offset;
        rtm_ext->extractors[j].num_elements = ext.num_elements;
      }
    }
    addr += sizeof(SNodeExtractors) * max_snodes;
    // init snode_lists
    ListManager *const rtm_list_head = reinterpret_cast<ListManager *>(addr);
    int list_data_mem_begin = 0;
    for (int i = 0; i < max_snodes; ++i) {
      auto iter = snode_descriptors.find(i);
      if (iter == snode_descriptors.end()) {
        continue;
      }
      const SNodeDescriptor &sn_meta = iter->second;
      ListManager *rtm_list = reinterpret_cast<ListManager *>(addr) + i;
      rtm_list->element_stride = sizeof(ListgenElement);
      // This can be really large, especially for other sparse SNodes (e.g.
      // dynamic, hash). In the future, Metal might also be able to support
      // dynamic memory allocation from the kernel side. That should help reduce
      // the initial size.
      rtm_list->max_num_elems = sn_meta.total_num_elems_from_root;
      rtm_list->next = 0;
      rtm_list->mem_begin = list_data_mem_begin;
      list_data_mem_begin += rtm_list->max_num_elems * rtm_list->element_stride;
      TI_DEBUG("ListManager\n  id={}\n  num_slots={}\n  mem_begin={}\n", i,
               rtm_list->max_num_elems, rtm_list->mem_begin);
    }
    addr += sizeof(ListManager) * max_snodes;
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
    // root list data are static
    ListgenElement root_elem;
    root_elem.root_mem_offset = 0;
    for (int i = 0; i < taichi_max_num_indices; ++i) {
      root_elem.coords[i] = 0;
    }
    append(rtm_list_head + root_id, root_elem, addr);
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
  ProfilerBase *const profiler_;
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
  std::unordered_map<std::string, std::unique_ptr<CompiledTaichiKernel>>
      compiled_taichi_kernels_;
};

#else

class KernelManager::Impl {
 public:
  explicit Impl(Params) {
    TI_ERROR("Metal not supported on the current OS");
  }

  void register_taichi_kernel(
      const std::string &taichi_kernel_name,
      const std::string &mtl_kernel_source_code,
      const std::vector<KernelAttributes> &kernels_attribs,
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
    const std::vector<KernelAttributes> &kernels_attribs,
    const KernelContextAttributes &ctx_attribs) {
  impl_->register_taichi_kernel(taichi_kernel_name, mtl_kernel_source_code,
                                kernels_attribs, ctx_attribs);
}

void KernelManager::launch_taichi_kernel(const std::string &taichi_kernel_name,
                                         Context *ctx) {
  impl_->launch_taichi_kernel(taichi_kernel_name, ctx);
}

void KernelManager::synchronize() {
  impl_->synchronize();
}

}  // namespace metal
TLANG_NAMESPACE_END
