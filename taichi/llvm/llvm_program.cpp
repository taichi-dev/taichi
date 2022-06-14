#include "taichi/llvm/llvm_program.h"

#include "llvm/IR/Module.h"

#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/arch.h"
#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/math/arithmetic.h"
#include "taichi/runtime/llvm/mem_request.h"
#include "taichi/util/str.h"
#include "taichi/codegen/codegen.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/backends/cpu/aot_module_builder_impl.h"
#include "taichi/backends/cpu/cpu_device.h"
#include "taichi/backends/cuda/cuda_device.h"

#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/aot_module_builder_impl.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/codegen_cuda.h"
#include "taichi/backends/cuda/cuda_context.h"
#endif

namespace taichi {
namespace lang {
namespace {
void assert_failed_host(const char *msg) {
  TI_ERROR("Assertion failure: {}", msg);
}

void *taichi_allocate_aligned(MemoryPool *memory_pool,
                              std::size_t size,
                              std::size_t alignment) {
  return memory_pool->allocate(size, alignment);
}
}  // namespace

LlvmProgramImpl::LlvmProgramImpl(CompileConfig &config_,
                                 KernelProfilerBase *profiler)
    : ProgramImpl(config_) {
  runtime_mem_info_ = Runtime::create(config_.arch);
  if (config_.arch == Arch::cuda) {
    if (!runtime_mem_info_) {
      TI_WARN("Taichi is not compiled with CUDA.");
      config_.arch = host_arch();
    } else if (!is_cuda_api_available()) {
      TI_WARN("No CUDA driver API detected.");
      config_.arch = host_arch();
    } else if (!runtime_mem_info_->detected()) {
      TI_WARN("No CUDA device detected.");
      config_.arch = host_arch();
    } else {
      // CUDA runtime created successfully
    }
    if (config_.arch != Arch::cuda) {
      TI_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  }

  snode_tree_buffer_manager_ = std::make_unique<SNodeTreeBufferManager>(this);

  thread_pool_ = std::make_unique<ThreadPool>(config->cpu_max_num_threads);

  preallocated_device_buffer_ = nullptr;
  llvm_runtime_ = nullptr;
  llvm_context_host_ = std::make_unique<TaichiLLVMContext>(this, host_arch());
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    int num_SMs{1};
    CUDADriver::get_instance().device_get_attribute(
        &num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
    int query_max_block_dim{1024};
    CUDADriver::get_instance().device_get_attribute(
        &query_max_block_dim, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, nullptr);
    int version{0};
    CUDADriver::get_instance().driver_get_version(&version);
    int query_max_block_per_sm{16};
    if (version >= 11000) {
      // query this attribute only when CUDA version is above 11.0
      CUDADriver::get_instance().device_get_attribute(
          &query_max_block_per_sm,
          CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
    }

    if (config_.max_block_dim == 0) {
      config_.max_block_dim = query_max_block_dim;
    }

    if (config_.saturating_grid_dim == 0) {
      if (version >= 11000) {
        TI_TRACE("CUDA max blocks per SM = {}", query_max_block_per_sm);
      }
      config_.saturating_grid_dim = num_SMs * query_max_block_per_sm * 2;
    }
#endif
  }

  if (arch_is_cpu(config->arch)) {
    config_.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();
  }

  if (config->kernel_profiler && runtime_mem_info_) {
    runtime_mem_info_->set_profiler(profiler);
  }
#if defined(TI_WITH_CUDA)
  if (config_.arch == Arch::cuda) {
    if (config_.kernel_profiler) {
      CUDAContext::get_instance().set_profiler(profiler);
    } else {
      CUDAContext::get_instance().set_profiler(nullptr);
    }
    CUDAContext::get_instance().set_debug(config->debug);
    device_ = std::make_shared<cuda::CudaDevice>();
  }
#endif
}

void LlvmProgramImpl::initialize_host() {
  // Note this cannot be placed inside LlvmProgramImpl constructor, see doc
  // string for init_runtime_jit_module() for more details.
  llvm_context_host_->init_runtime_jit_module();
}

void LlvmProgramImpl::maybe_initialize_cuda_llvm_context() {
  if (config->arch == Arch::cuda && llvm_context_device_ == nullptr) {
    llvm_context_device_ =
        std::make_unique<TaichiLLVMContext>(this, Arch::cuda);
    llvm_context_device_->init_runtime_jit_module();
  }
}

FunctionType LlvmProgramImpl::compile(Kernel *kernel,
                                      OffloadedStmt *offloaded) {
  auto codegen = KernelCodeGen::create(kernel->arch, kernel, offloaded);
  return codegen->codegen();
}

void LlvmProgramImpl::synchronize() {
  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    TI_ERROR("No CUDA support");
#endif
  }
}

std::unique_ptr<llvm::Module>
LlvmProgramImpl::clone_struct_compiler_initial_context(
    bool has_multiple_snode_trees,
    TaichiLLVMContext *tlctx) {
  if (has_multiple_snode_trees) {
    return tlctx->clone_struct_module();
  }
  return tlctx->clone_runtime_module();
}

void LlvmProgramImpl::initialize_llvm_runtime_snodes(
    const LlvmOfflineCache::FieldCacheData &field_cache_data,
    uint64 *result_buffer) {
  TaichiLLVMContext *tlctx = nullptr;
  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    tlctx = llvm_context_device_.get();
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    tlctx = llvm_context_host_.get();
  }

  auto *const runtime_jit = tlctx->runtime_jit_module;
  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  size_t root_size = field_cache_data.root_size;
  const auto snode_metas = field_cache_data.snode_metas;
  const int tree_id = field_cache_data.tree_id;
  const int root_id = field_cache_data.root_id;

  TI_TRACE("Allocating data structure of size {} bytes", root_size);
  std::size_t rounded_size = taichi::iroundup(root_size, taichi_page_size);

  Ptr root_buffer = snode_tree_buffer_manager_->allocate(
      runtime_jit, llvm_runtime_, rounded_size, taichi_page_size, tree_id,
      result_buffer);
  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    std::memset(root_buffer, 0, rounded_size);
  }

  DeviceAllocation alloc{kDeviceNullAllocation};

  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    alloc = cuda_device()->import_memory(root_buffer, rounded_size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    alloc = cpu_device()->import_memory(root_buffer, rounded_size);
  }

  snode_tree_allocs_[tree_id] = alloc;

  bool all_dense = config->demote_dense_struct_fors;
  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (snode_metas[i].type != SNodeType::dense &&
        snode_metas[i].type != SNodeType::place &&
        snode_metas[i].type != SNodeType::root) {
      all_dense = false;
      break;
    }
  }

  runtime_jit->call<void *, std::size_t, int, int, int, std::size_t, Ptr>(
      "runtime_initialize_snodes", llvm_runtime_, root_size, root_id,
      (int)snode_metas.size(), tree_id, rounded_size, root_buffer, all_dense);

  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (is_gc_able(snode_metas[i].type)) {
      const auto snode_id = snode_metas[i].id;
      std::size_t node_size;
      auto element_size = snode_metas[i].cell_size_bytes;
      if (snode_metas[i].type == SNodeType::pointer) {
        // pointer. Allocators are for single elements
        node_size = element_size;
      } else {
        // dynamic. Allocators are for the chunks
        node_size = sizeof(void *) + element_size * snode_metas[i].chunk_size;
      }
      TI_TRACE("Initializing allocator for snode {} (node size {})", snode_id,
               node_size);
      auto rt = llvm_runtime_;
      runtime_jit->call<void *, int, std::size_t>(
          "runtime_NodeAllocator_initialize", rt, snode_id, node_size);
      TI_TRACE("Allocating ambient element for snode {} (node size {})",
               snode_id, node_size);
      runtime_jit->call<void *, int>("runtime_allocate_ambient", rt, snode_id,
                                     node_size);
    }
  }
}

std::unique_ptr<StructCompiler> LlvmProgramImpl::compile_snode_tree_types_impl(
    SNodeTree *tree) {
  auto *const root = tree->root();
  const bool has_multiple_snode_trees = (num_snode_trees_processed_ > 0);
  std::unique_ptr<StructCompiler> struct_compiler{nullptr};
  if (arch_is_cpu(config->arch)) {
    auto host_module = clone_struct_compiler_initial_context(
        has_multiple_snode_trees, llvm_context_host_.get());
    struct_compiler = std::make_unique<StructCompilerLLVM>(
        host_arch(), this, std::move(host_module), tree->id());

  } else {
    TI_ASSERT(config->arch == Arch::cuda);
    auto device_module = clone_struct_compiler_initial_context(
        has_multiple_snode_trees, llvm_context_device_.get());
    struct_compiler = std::make_unique<StructCompilerLLVM>(
        Arch::cuda, this, std::move(device_module), tree->id());
  }
  struct_compiler->run(*root);
  ++num_snode_trees_processed_;
  return struct_compiler;
}

void LlvmProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  auto struct_compiler = compile_snode_tree_types_impl(tree);
  int snode_tree_id = tree->id();
  int root_id = tree->root()->id;

  // Add compiled result to Cache
  cache_field(snode_tree_id, root_id, *struct_compiler);
}

void LlvmProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                             uint64 *result_buffer) {
  compile_snode_tree_types(tree);
  int snode_tree_id = tree->id();

  TI_ASSERT(cache_data_.fields.find(snode_tree_id) != cache_data_.fields.end());
  initialize_llvm_runtime_snodes(cache_data_.fields.at(snode_tree_id),
                                 result_buffer);
}

uint64 LlvmProgramImpl::fetch_result_uint64(int i, uint64 *result_buffer) {
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  synchronize();
  uint64 ret;
  auto arch = config->arch;
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                     sizeof(uint64));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    ret = result_buffer[i];
  }
  return ret;
}

std::size_t LlvmProgramImpl::get_snode_num_dynamically_allocated(
    SNode *snode,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config->arch));

  auto node_allocator =
      runtime_query<void *>("LLVMRuntime_get_node_allocators", result_buffer,
                            llvm_runtime_, snode->id);
  auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                         result_buffer, node_allocator);

  return (std::size_t)runtime_query<int32>("ListManager_get_num_elements",
                                           result_buffer, data_list);
}

void LlvmProgramImpl::print_list_manager_info(void *list_manager,
                                              uint64 *result_buffer) {
  auto list_manager_len = runtime_query<int32>("ListManager_get_num_elements",
                                               result_buffer, list_manager);

  auto element_size = runtime_query<int32>("ListManager_get_element_size",
                                           result_buffer, list_manager);

  auto elements_per_chunk =
      runtime_query<int32>("ListManager_get_max_num_elements_per_chunk",
                           result_buffer, list_manager);

  auto num_active_chunks = runtime_query<int32>(
      "ListManager_get_num_active_chunks", result_buffer, list_manager);

  auto size_MB = 1e-6f * num_active_chunks * elements_per_chunk * element_size;

  fmt::print(
      " length={:n}     {:n} chunks x [{:n} x {:n} B]  total={:.4f} MB\n",
      list_manager_len, num_active_chunks, elements_per_chunk, element_size,
      size_MB);
}

std::unique_ptr<AotModuleBuilder> LlvmProgramImpl::make_aot_module_builder() {
  if (config->arch == Arch::x64 || config->arch == Arch::arm64) {
    return std::make_unique<cpu::AotModuleBuilderImpl>(this);
  }

#if defined(TI_WITH_CUDA)
  if (config->arch == Arch::cuda) {
    return std::make_unique<cuda::AotModuleBuilderImpl>(this);
  }
#endif

  TI_NOT_IMPLEMENTED;
  return nullptr;
}

void LlvmProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  maybe_initialize_cuda_llvm_context();

  std::size_t prealloc_size = 0;
  TaichiLLVMContext *tlctx = nullptr;
  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().malloc(
        (void **)result_buffer_ptr,
        sizeof(uint64) * taichi_result_buffer_entries);
    const auto total_mem = runtime_mem_info_->get_total_memory();
    if (config->device_memory_fraction == 0) {
      TI_ASSERT(config->device_memory_GB > 0);
      prealloc_size = std::size_t(config->device_memory_GB * (1UL << 30));
    } else {
      prealloc_size = std::size_t(config->device_memory_fraction * total_mem);
    }
    TI_ASSERT(prealloc_size <= total_mem);

    TI_TRACE("Allocating device memory {:.2f} GB",
             1.0 * prealloc_size / (1UL << 30));

    Device::AllocParams preallocated_device_buffer_alloc_params;
    preallocated_device_buffer_alloc_params.size = prealloc_size;
    preallocated_device_buffer_alloc_ =
        cuda_device()->allocate_memory(preallocated_device_buffer_alloc_params);
    cuda::CudaDevice::AllocInfo preallocated_device_buffer_alloc_info =
        cuda_device()->get_alloc_info(preallocated_device_buffer_alloc_);
    preallocated_device_buffer_ = preallocated_device_buffer_alloc_info.ptr;

    CUDADriver::get_instance().memset(preallocated_device_buffer_, 0,
                                      prealloc_size);
    tlctx = llvm_context_device_.get();
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    *result_buffer_ptr = (uint64 *)memory_pool->allocate(
        sizeof(uint64) * taichi_result_buffer_entries, 8);
    tlctx = llvm_context_host_.get();
  }
  auto *const runtime_jit = tlctx->runtime_jit_module;

  // Starting random state for the program calculated using the random seed.
  // The seed is multiplied by 2^20 so that two programs with different seeds
  // will not have overlapping random states in any thread.
  int starting_rand_state = config->random_seed * 1048576;

  // Number of random states. One per CPU/CUDA thread.
  int num_rand_states = 0;

  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    // It is important to make sure that every CUDA thread has its own random
    // state so that we do not need expensive per-state locks.
    num_rand_states = config->saturating_grid_dim * config->max_block_dim;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    num_rand_states = config->cpu_max_num_threads;
  }

  TI_TRACE("Allocating {} random states (used by CUDA only)", num_rand_states);

  runtime_jit->call<void *, void *, std::size_t, void *, int, int, void *,
                    void *, void *>(
      "runtime_initialize", *result_buffer_ptr, memory_pool, prealloc_size,
      preallocated_device_buffer_, starting_rand_state, num_rand_states,
      (void *)&taichi_allocate_aligned, (void *)std::printf,
      (void *)std::vsnprintf);

  TI_TRACE("LLVMRuntime initialized (excluding `root`)");
  llvm_runtime_ = fetch_result<void *>(taichi_result_buffer_ret_value_id,
                                       *result_buffer_ptr);
  TI_TRACE("LLVMRuntime pointer fetched");

  if (arch_use_host_memory(config->arch)) {
    runtime_jit->call<void *>("runtime_get_mem_req_queue", llvm_runtime_);
    auto mem_req_queue = fetch_result<void *>(taichi_result_buffer_ret_value_id,
                                              *result_buffer_ptr);
    memory_pool->set_queue((MemRequestQueue *)mem_req_queue);
  }

  if (arch_use_host_memory(config->arch)) {
    runtime_jit->call<void *, void *, void *>(
        "LLVMRuntime_initialize_thread_pool", llvm_runtime_, thread_pool_.get(),
        (void *)ThreadPool::static_run);

    runtime_jit->call<void *, void *>("LLVMRuntime_set_assert_failed",
                                      llvm_runtime_,
                                      (void *)assert_failed_host);
  }
  if (arch_is_cpu(config->arch) && (profiler != nullptr)) {
    // Profiler functions can only be called on CPU kernels
    runtime_jit->call<void *, void *>("LLVMRuntime_set_profiler", llvm_runtime_,
                                      profiler);
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_profiler_start", llvm_runtime_,
        (void *)&KernelProfilerBase::profiler_start);
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_profiler_stop", llvm_runtime_,
        (void *)&KernelProfilerBase::profiler_stop);
  }
}

void LlvmProgramImpl::check_runtime_error(uint64 *result_buffer) {
  synchronize();
  auto tlctx = llvm_context_host_.get();
  if (llvm_context_device_) {
    // In case there is a standalone device context (e.g. CUDA without unified
    // memory), use the device context instead.
    tlctx = llvm_context_device_.get();
  }
  auto *runtime_jit_module = tlctx->runtime_jit_module;
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_error_code",
                                   llvm_runtime_);
  auto error_code =
      fetch_result<int64>(taichi_result_buffer_error_id, result_buffer);

  if (error_code) {
    std::string error_message_template;

    // Here we fetch the error_message_template char by char.
    // This is not efficient, but fortunately we only need to do this when an
    // assertion fails. Note that we may not have unified memory here, so using
    // "fetch_result" that works across device/host memroy is necessary.
    for (int i = 0;; i++) {
      runtime_jit_module->call<void *>("runtime_retrieve_error_message",
                                       llvm_runtime_, i);
      auto c = fetch_result<char>(taichi_result_buffer_error_id, result_buffer);
      error_message_template += c;
      if (c == '\0') {
        break;
      }
    }

    if (error_code == 1) {
      const auto error_message_formatted = format_error_message(
          error_message_template,
          [runtime_jit_module, result_buffer, this](int argument_id) {
            runtime_jit_module->call<void *>(
                "runtime_retrieve_error_message_argument", llvm_runtime_,
                argument_id);
            return fetch_result<uint64>(taichi_result_buffer_error_id,
                                        result_buffer);
          });
      throw TaichiAssertionError(error_message_formatted);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }
}

void LlvmProgramImpl::finalize() {
  if (runtime_mem_info_)
    runtime_mem_info_->set_profiler(nullptr);
#if defined(TI_WITH_CUDA)
  if (preallocated_device_buffer_ != nullptr) {
    cuda_device()->dealloc_memory(preallocated_device_buffer_alloc_);
  }
#endif
}

void LlvmProgramImpl::print_memory_profiler_info(
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config->arch));

  fmt::print("\n[Memory Profiler]\n");

  std::locale::global(std::locale("en_US.UTF-8"));
  // So that thousand separators are added to "{:n}" slots in fmtlib.
  // E.g., 10000 is printed as "10,000".
  // TODO: is there a way to set locale only locally in this function?

  std::function<void(SNode *, int)> visit = [&](SNode *snode, int depth) {
    auto element_list =
        runtime_query<void *>("LLVMRuntime_get_element_lists", result_buffer,
                              llvm_runtime_, snode->id);

    if (snode->type != SNodeType::place) {
      fmt::print("SNode {:10}\n", snode->get_node_type_name_hinted());

      if (element_list) {
        fmt::print("  active element list:");
        print_list_manager_info(element_list, result_buffer);

        auto node_allocator =
            runtime_query<void *>("LLVMRuntime_get_node_allocators",
                                  result_buffer, llvm_runtime_, snode->id);

        if (node_allocator) {
          auto free_list = runtime_query<void *>("NodeManager_get_free_list",
                                                 result_buffer, node_allocator);
          auto recycled_list = runtime_query<void *>(
              "NodeManager_get_recycled_list", result_buffer, node_allocator);

          auto free_list_len = runtime_query<int32>(
              "ListManager_get_num_elements", result_buffer, free_list);

          auto recycled_list_len = runtime_query<int32>(
              "ListManager_get_num_elements", result_buffer, recycled_list);

          auto free_list_used = runtime_query<int32>(
              "NodeManager_get_free_list_used", result_buffer, node_allocator);

          auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                                 result_buffer, node_allocator);
          fmt::print("  data list:          ");
          print_list_manager_info(data_list, result_buffer);

          fmt::print(
              "  Allocated elements={:n}; free list length={:n}; recycled list "
              "length={:n}\n",
              free_list_used, free_list_len, recycled_list_len);
        }
      }
    }
    for (const auto &ch : snode->ch) {
      visit(ch.get(), depth + 1);
    }
  };

  for (auto &a : snode_trees_) {
    visit(a->root(), /*depth=*/0);
  }

  auto total_requested_memory = runtime_query<std::size_t>(
      "LLVMRuntime_get_total_requested_memory", result_buffer, llvm_runtime_);

  fmt::print(
      "Total requested dynamic memory (excluding alignment padding): {:n} B\n",
      total_requested_memory);
}

cuda::CudaDevice *LlvmProgramImpl::cuda_device() {
  if (config->arch != Arch::cuda) {
    TI_ERROR("arch is not cuda");
  }
  return static_cast<cuda::CudaDevice *>(device_.get());
}

cpu::CpuDevice *LlvmProgramImpl::cpu_device() {
  TI_ERROR_IF(!arch_is_cpu(config->arch), "arch is not cpu");
  return static_cast<cpu::CpuDevice *>(device_.get());
}

LlvmDevice *LlvmProgramImpl::llvm_device() {
  TI_ASSERT(dynamic_cast<LlvmDevice *>(device_.get()));
  return static_cast<LlvmDevice *>(device_.get());
}

DevicePtr LlvmProgramImpl::get_snode_tree_device_ptr(int tree_id) {
  DeviceAllocation tree_alloc = snode_tree_allocs_[tree_id];
  return tree_alloc.get_ptr();
}

DeviceAllocation LlvmProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  TaichiLLVMContext *tlctx = nullptr;
  if (llvm_context_device_) {
    tlctx = llvm_context_device_.get();
  } else {
    tlctx = llvm_context_host_.get();
  }

  return llvm_device()->allocate_memory_runtime(
      {{alloc_size, /*host_write=*/false, /*host_read=*/false,
        /*export_sharing=*/false, AllocUsage::Storage},
       config->ndarray_use_cached_allocator,
       tlctx->runtime_jit_module,
       get_llvm_runtime(),
       result_buffer});
}

uint64_t *LlvmProgramImpl::get_ndarray_alloc_info_ptr(
    const DeviceAllocation &alloc) {
  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return (uint64_t *)cuda_device()->get_alloc_info(alloc).ptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    return (uint64_t *)cpu_device()->get_alloc_info(alloc).ptr;
  }
}

void LlvmProgramImpl::fill_ndarray(const DeviceAllocation &alloc,
                                   std::size_t size,
                                   uint32_t data) {
  auto ptr = get_ndarray_alloc_info_ptr(alloc);
  if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memsetd32((void *)ptr, data, size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    std::fill((uint32_t *)ptr, (uint32_t *)ptr + size, data);
  }
}

void LlvmProgramImpl::cache_kernel(
    const std::string &kernel_key,
    llvm::Module *module,
    std::vector<LlvmLaunchArgInfo> &&args,
    std::vector<LlvmOfflineCache::OffloadedTaskCacheData>
        &&offloaded_task_list) {
  if (cache_data_.kernels.find(kernel_key) != cache_data_.kernels.end()) {
    return;
  }
  auto &kernel_cache = cache_data_.kernels[kernel_key];
  kernel_cache.kernel_key = kernel_key;
  kernel_cache.owned_module = llvm::CloneModule(*module);
  kernel_cache.args = std::move(args);
  kernel_cache.offloaded_task_list = std::move(offloaded_task_list);
}

void LlvmProgramImpl::cache_field(int snode_tree_id,
                                  int root_id,
                                  const StructCompiler &struct_compiler) {
  if (cache_data_.fields.find(snode_tree_id) != cache_data_.fields.end()) {
    // [TODO] check and update the Cache, instead of simply return.
    return;
  }

  LlvmOfflineCache::FieldCacheData ret;
  ret.tree_id = snode_tree_id;
  ret.root_id = root_id;
  ret.root_size = struct_compiler.root_size;

  const auto &snodes = struct_compiler.snodes;
  for (size_t i = 0; i < snodes.size(); i++) {
    LlvmOfflineCache::FieldCacheData::SNodeCacheData snode_cache_data;
    snode_cache_data.id = snodes[i]->id;
    snode_cache_data.type = snodes[i]->type;
    snode_cache_data.cell_size_bytes = snodes[i]->cell_size_bytes;
    snode_cache_data.chunk_size = snodes[i]->chunk_size;

    ret.snode_metas.emplace_back(std::move(snode_cache_data));
  }

  cache_data_.fields[snode_tree_id] = std::move(ret);
}

void LlvmProgramImpl::dump_cache_data_to_disk() {
  if (config->offline_cache && !cache_data_.kernels.empty()) {
    LlvmOfflineCacheFileWriter writer{};
    writer.set_data(std::move(cache_data_));
    writer.dump(config->offline_cache_file_path);
  }
}

}  // namespace lang
}  // namespace taichi
