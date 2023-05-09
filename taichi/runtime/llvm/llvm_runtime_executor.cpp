#include "taichi/runtime/llvm/llvm_runtime_executor.h"

#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/llvm/device_memory_pool.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

#include "taichi/platform/amdgpu/detect_amdgpu.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_device.h"
#if defined(TI_WITH_AMDGPU)
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#endif

namespace taichi::lang {
namespace {
void assert_failed_host(const char *msg) {
  TI_ERROR("Assertion failure: {}", msg);
}

void *host_allocate_aligned(HostMemoryPool *memory_pool,
                            std::size_t size,
                            std::size_t alignment) {
  return memory_pool->allocate(size, alignment);
}

}  // namespace

LlvmRuntimeExecutor::LlvmRuntimeExecutor(CompileConfig &config,
                                         KernelProfilerBase *profiler)
    : config_(config) {
  if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    if (!is_cuda_api_available()) {
      TI_WARN("No CUDA driver API detected.");
      config.arch = host_arch();
    } else if (!CUDAContext::get_instance().detected()) {
      TI_WARN("No CUDA device detected.");
      config.arch = host_arch();
    } else {
      // CUDA runtime created successfully
    }
#else
    TI_WARN("Taichi is not compiled with CUDA.");
    config.arch = host_arch();
#endif

    if (config.arch != Arch::cuda) {
      TI_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  } else if (config.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    if (!is_rocm_api_available()) {
      TI_WARN("No AMDGPU ROCm API detected.");
      config.arch = host_arch();
    } else if (!AMDGPUContext::get_instance().detected()) {
      TI_WARN("No AMDGPU device detected.");
      config.arch = host_arch();
    } else {
      // AMDGPU runtime created successfully
    }
#else
    TI_WARN("Taichi is not compiled with AMDGPU.");
    config.arch = host_arch();
#endif
  }

  if (config.kernel_profiler) {
    profiler_ = profiler;
  }

  snode_tree_buffer_manager_ = std::make_unique<SNodeTreeBufferManager>(this);
  thread_pool_ = std::make_unique<ThreadPool>(config.cpu_max_num_threads);
  preallocated_device_buffer_ = nullptr;

  llvm_runtime_ = nullptr;

  if (arch_is_cpu(config.arch)) {
    config.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();

  }
#if defined(TI_WITH_CUDA)
  else if (config.arch == Arch::cuda) {
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

    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }

    if (config.saturating_grid_dim == 0) {
      if (version >= 11000) {
        TI_TRACE("CUDA max blocks per SM = {}", query_max_block_per_sm);
      }
      config.saturating_grid_dim = num_SMs * query_max_block_per_sm * 2;
    }
    if (config.kernel_profiler) {
      CUDAContext::get_instance().set_profiler(profiler);
    } else {
      CUDAContext::get_instance().set_profiler(nullptr);
    }
    CUDAContext::get_instance().set_debug(config.debug);
    if (config.cuda_stack_limit != 0) {
      CUDADriver::get_instance().context_set_limit(CU_LIMIT_STACK_SIZE,
                                                   config.cuda_stack_limit);
    }
    device_ = std::make_shared<cuda::CudaDevice>();
  }
#endif
#if defined(TI_WITH_AMDGPU)
  else if (config.arch == Arch::amdgpu) {
    int num_workgroups{1};
    AMDGPUDriver::get_instance().device_get_attribute(
        &num_workgroups, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
    int query_max_block_dim{1024};
    AMDGPUDriver::get_instance().device_get_attribute(
        &query_max_block_dim, HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);
    // magic number 32
    // I didn't find the relevant parameter to limit the max block num per CU
    // So ....
    int query_max_block_per_cu{32};
    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }
    if (config.saturating_grid_dim == 0) {
      config.saturating_grid_dim = num_workgroups * query_max_block_per_cu * 2;
    }
    if (config.kernel_profiler) {
      AMDGPUContext::get_instance().set_profiler(profiler);
    } else {
      AMDGPUContext::get_instance().set_profiler(nullptr);
    }
    AMDGPUContext::get_instance().set_debug(config.debug);
    device_ = std::make_shared<amdgpu::AmdgpuDevice>();
  }
#endif
#ifdef TI_WITH_DX12
  else if (config.arch == Arch::dx12) {
    // FIXME: add dx12 device.
    // FIXME: set value based on DX12.
    config.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();
  }
#endif
  else {
    TI_NOT_IMPLEMENTED
  }
  llvm_context_ = std::make_unique<TaichiLLVMContext>(
      config_, arch_is_cpu(config.arch) ? host_arch() : config.arch);
  jit_session_ = JITSession::create(llvm_context_.get(), config, config.arch);
  init_runtime_jit_module(llvm_context_->clone_runtime_module());
}

TaichiLLVMContext *LlvmRuntimeExecutor::get_llvm_context() {
  return llvm_context_.get();
}

JITModule *LlvmRuntimeExecutor::create_jit_module(
    std::unique_ptr<llvm::Module> module) {
  return jit_session_->add_module(std::move(module));
}

JITModule *LlvmRuntimeExecutor::get_runtime_jit_module() {
  return runtime_jit_module_;
}

void LlvmRuntimeExecutor::print_list_manager_info(void *list_manager,
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

void LlvmRuntimeExecutor::synchronize() {
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    TI_ERROR("No CUDA support");
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    // A better way
    // use `hipFreeAsync` to free the device kernel arg mem
    // notice: rocm version
    AMDGPUContext::get_instance().free_kernel_arg_pointer();
#else
    TI_ERROR("No AMDGPU support");
#endif
  }
  fflush(stdout);
}

uint64 LlvmRuntimeExecutor::fetch_result_uint64(int i, uint64 *result_buffer) {
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  synchronize();
  uint64 ret;
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                     sizeof(uint64));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                       sizeof(uint64));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    ret = result_buffer[i];
  }
  return ret;
}

std::size_t LlvmRuntimeExecutor::get_snode_num_dynamically_allocated(
    SNode *snode,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config_.arch));

  auto node_allocator =
      runtime_query<void *>("LLVMRuntime_get_node_allocators", result_buffer,
                            llvm_runtime_, snode->id);
  auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                         result_buffer, node_allocator);

  return (std::size_t)runtime_query<int32>("ListManager_get_num_elements",
                                           result_buffer, data_list);
}

void LlvmRuntimeExecutor::check_runtime_error(uint64 *result_buffer) {
  synchronize();
  auto *runtime_jit_module = get_runtime_jit_module();
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_error_code",
                                   llvm_runtime_);
  auto error_code =
      fetch_result<int64>(taichi_result_buffer_error_id, result_buffer);

  if (error_code) {
    std::string error_message_template;

    // Here we fetch the error_message_template char by char.
    // This is not efficient, but fortunately we only need to do this when an
    // assertion fails. Note that we may not have unified memory here, so using
    // "fetch_result" that works across device/host memory is necessary.
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

void LlvmRuntimeExecutor::print_memory_profiler_info(
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config_.arch));

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

DevicePtr LlvmRuntimeExecutor::get_snode_tree_device_ptr(int tree_id) {
  DeviceAllocation tree_alloc = snode_tree_allocs_[tree_id];
  return tree_alloc.get_ptr();
}

void LlvmRuntimeExecutor::initialize_llvm_runtime_snodes(
    const LlvmOfflineCache::FieldCacheData &field_cache_data,
    uint64 *result_buffer) {
  auto *const runtime_jit = get_runtime_jit_module();
  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  size_t root_size = field_cache_data.root_size;
  const auto snode_metas = field_cache_data.snode_metas;
  const int tree_id = field_cache_data.tree_id;
  const int root_id = field_cache_data.root_id;

  TI_TRACE("Allocating data structure of size {} bytes", root_size);
  std::size_t rounded_size = taichi::iroundup(root_size, taichi_page_size);

  Ptr root_buffer = snode_tree_buffer_manager_->allocate(rounded_size, tree_id,
                                                         result_buffer);
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    std::memset(root_buffer, 0, rounded_size);
  }

  DeviceAllocation alloc =
      llvm_device()->import_memory(root_buffer, rounded_size);

  snode_tree_allocs_[tree_id] = alloc;

  bool all_dense = config_.demote_dense_struct_fors;
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
      runtime_jit->call<void *, int, std::size_t>(
          "runtime_NodeAllocator_initialize", llvm_runtime_, snode_id,
          node_size);
      TI_TRACE("Allocating ambient element for snode {} (node size {})",
               snode_id, node_size);
      runtime_jit->call<void *, int>("runtime_allocate_ambient", llvm_runtime_,
                                     snode_id, node_size);
    }
  }
}

LlvmDevice *LlvmRuntimeExecutor::llvm_device() {
  TI_ASSERT(dynamic_cast<LlvmDevice *>(device_.get()));
  return static_cast<LlvmDevice *>(device_.get());
}

DeviceAllocation LlvmRuntimeExecutor::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  return llvm_device()->allocate_memory_runtime(
      {{alloc_size, /*host_write=*/false, /*host_read=*/false,
        /*export_sharing=*/false, AllocUsage::Storage},
       get_runtime_jit_module(),
       get_llvm_runtime(),
       result_buffer});
}

void LlvmRuntimeExecutor::deallocate_memory_ndarray(DeviceAllocation handle) {
  llvm_device()->dealloc_memory(handle);
}

void LlvmRuntimeExecutor::fill_ndarray(const DeviceAllocation &alloc,
                                       std::size_t size,
                                       uint32_t data) {
  auto ptr = get_ndarray_alloc_info_ptr(alloc);
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memsetd32((void *)ptr, data, size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memset((void *)ptr, data, size);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    std::fill((uint32_t *)ptr, (uint32_t *)ptr + size, data);
  }
}

uint64_t *LlvmRuntimeExecutor::get_ndarray_alloc_info_ptr(
    const DeviceAllocation &alloc) {
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return (uint64_t *)llvm_device()
        ->as<cuda::CudaDevice>()
        ->get_alloc_info(alloc)
        .ptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    return (uint64_t *)llvm_device()
        ->as<amdgpu::AmdgpuDevice>()
        ->get_alloc_info(alloc)
        .ptr;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  return (uint64_t *)llvm_device()
      ->as<cpu::CpuDevice>()
      ->get_alloc_info(alloc)
      .ptr;
}

void LlvmRuntimeExecutor::finalize() {
  profiler_ = nullptr;
  if (preallocated_device_buffer_ != nullptr) {
    if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
      llvm_device()->dealloc_memory(preallocated_device_buffer_alloc_);
      llvm_device()->clear();
      DeviceMemoryPool::get_instance().reset();
    }
  }
  finalized_ = true;
}

LlvmRuntimeExecutor::~LlvmRuntimeExecutor() {
  if (!finalized_) {
    finalize();
  }
}

void LlvmRuntimeExecutor::materialize_runtime(KernelProfilerBase *profiler,
                                              uint64 **result_buffer_ptr) {
  // The result buffer allocated here is only used for the launches of
  // runtime JIT functions. To avoid memory leak, we use the head of
  // the preallocated device buffer as the result buffer in
  // CUDA and AMDGPU backends.
  // | ==================preallocated device buffer ========================== |
  // |<- reserved for return ->|<---- usable for allocators on the device ---->|

  std::size_t prealloc_size = 0;
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    const auto total_mem = CUDAContext::get_instance().get_total_memory();
    if (config_.device_memory_fraction == 0) {
      TI_ASSERT(config_.device_memory_GB > 0);
      prealloc_size = std::size_t(config_.device_memory_GB * (1UL << 30));
    } else {
      prealloc_size = std::size_t(config_.device_memory_fraction * total_mem);
    }
    TI_ASSERT(prealloc_size <= total_mem);

    TI_TRACE("Allocating device memory {:.2f} GB",
             1.0 * prealloc_size / (1UL << 30));

    Device::AllocParams preallocated_device_buffer_alloc_params;
    preallocated_device_buffer_alloc_params.size = prealloc_size;
    RhiResult res =
        llvm_device()->allocate_memory(preallocated_device_buffer_alloc_params,
                                       &preallocated_device_buffer_alloc_);
    TI_ASSERT(res == RhiResult::success);
    cuda::CudaDevice::AllocInfo preallocated_device_buffer_alloc_info =
        llvm_device()->as<cuda::CudaDevice>()->get_alloc_info(
            preallocated_device_buffer_alloc_);
    preallocated_device_buffer_ = preallocated_device_buffer_alloc_info.ptr;

    CUDADriver::get_instance().memset(preallocated_device_buffer_, 0,
                                      prealloc_size);
    *result_buffer_ptr = (uint64 *)preallocated_device_buffer_;
    size_t result_buffer_size = sizeof(uint64) * taichi_result_buffer_entries;
    preallocated_device_buffer_ =
        (char *)preallocated_device_buffer_ + result_buffer_size;
    prealloc_size -= result_buffer_size;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    const auto total_mem = AMDGPUContext::get_instance().get_total_memory();
    if (config_.device_memory_fraction == 0) {
      TI_ASSERT(config_.device_memory_GB > 0);
      prealloc_size = std::size_t(config_.device_memory_GB * (1UL << 30));
    } else {
      prealloc_size = std::size_t(config_.device_memory_fraction * total_mem);
    }
    TI_ASSERT(prealloc_size <= total_mem);

    TI_TRACE("Allocating device memory {:.2f} GB",
             1.0 * prealloc_size / (1UL << 30));

    Device::AllocParams preallocated_device_buffer_alloc_params;
    preallocated_device_buffer_alloc_params.size = prealloc_size;
    RhiResult res =
        llvm_device()->allocate_memory(preallocated_device_buffer_alloc_params,
                                       &preallocated_device_buffer_alloc_);
    TI_ASSERT(res == RhiResult::success);
    amdgpu::AmdgpuDevice::AllocInfo preallocated_device_buffer_alloc_info =
        llvm_device()->as<amdgpu::AmdgpuDevice>()->get_alloc_info(
            preallocated_device_buffer_alloc_);
    preallocated_device_buffer_ = preallocated_device_buffer_alloc_info.ptr;

    AMDGPUDriver::get_instance().memset(preallocated_device_buffer_, 0,
                                        prealloc_size);
    *result_buffer_ptr = (uint64 *)preallocated_device_buffer_;
    size_t result_buffer_size = sizeof(uint64) * taichi_result_buffer_entries;
    preallocated_device_buffer_ =
        (char *)preallocated_device_buffer_ + result_buffer_size;
    prealloc_size -= result_buffer_size;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    *result_buffer_ptr = (uint64 *)HostMemoryPool::get_instance().allocate(
        sizeof(uint64) * taichi_result_buffer_entries, 8);
  }
  auto *const runtime_jit = get_runtime_jit_module();

  // Starting random state for the program calculated using the random seed.
  // The seed is multiplied by 1048391 so that two programs with different seeds
  // will not have overlapping random states in any thread.
  int starting_rand_state = config_.random_seed * 1048391;

  // Number of random states. One per CPU/CUDA thread.
  int num_rand_states = 0;

  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_CUDA) || defined(TI_WITH_AMDGPU)
    // It is important to make sure that every CUDA thread has its own random
    // state so that we do not need expensive per-state locks.
    num_rand_states = config_.saturating_grid_dim * config_.max_block_dim;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    num_rand_states = config_.cpu_max_num_threads;
  }

  TI_TRACE("Launching runtime_initialize");

  auto *host_memory_pool = &HostMemoryPool::get_instance();
  runtime_jit
      ->call<void *, void *, std::size_t, void *, int, void *, void *, void *>(
          "runtime_initialize", *result_buffer_ptr, host_memory_pool,
          prealloc_size, preallocated_device_buffer_, num_rand_states,
          (void *)&host_allocate_aligned, (void *)std::printf,
          (void *)std::vsnprintf);

  TI_TRACE("LLVMRuntime initialized (excluding `root`)");
  llvm_runtime_ = fetch_result<void *>(taichi_result_buffer_ret_value_id,
                                       *result_buffer_ptr);
  TI_TRACE("LLVMRuntime pointer fetched");

  if (config_.arch == Arch::cuda) {
    TI_TRACE("Initializing {} random states using CUDA", num_rand_states);
    runtime_jit->launch<void *, int>(
        "runtime_initialize_rand_states_cuda", config_.saturating_grid_dim,
        config_.max_block_dim, 0, llvm_runtime_, starting_rand_state);
  } else {
    TI_TRACE("Initializing {} random states (serially)", num_rand_states);
    runtime_jit->call<void *, int>("runtime_initialize_rand_states_serial",
                                   llvm_runtime_, starting_rand_state);
  }

  if (arch_use_host_memory(config_.arch)) {
    runtime_jit->call<void *, void *, void *>(
        "LLVMRuntime_initialize_thread_pool", llvm_runtime_, thread_pool_.get(),
        (void *)ThreadPool::static_run);

    runtime_jit->call<void *, void *>("LLVMRuntime_set_assert_failed",
                                      llvm_runtime_,
                                      (void *)assert_failed_host);
  }
  if (arch_is_cpu(config_.arch) && (profiler != nullptr)) {
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
  if (arch_is_cpu(config_.arch) || config_.arch == Arch::cuda) {
    runtime_jit->call<void *>("runtime_initialize_runtime_context_buffer",
                              llvm_runtime_);
  }
}

void LlvmRuntimeExecutor::destroy_snode_tree(SNodeTree *snode_tree) {
  get_llvm_context()->delete_snode_tree(snode_tree->id());
  snode_tree_buffer_manager_->destroy(snode_tree);
}

Device *LlvmRuntimeExecutor::get_compute_device() {
  return device_.get();
}

LLVMRuntime *LlvmRuntimeExecutor::get_llvm_runtime() {
  return static_cast<LLVMRuntime *>(llvm_runtime_);
}

void LlvmRuntimeExecutor::init_runtime_jit_module(
    std::unique_ptr<llvm::Module> module) {
  llvm_context_->init_runtime_module(module.get());
  runtime_jit_module_ = create_jit_module(std::move(module));
}

}  // namespace taichi::lang
