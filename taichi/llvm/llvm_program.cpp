#include "llvm_program.h"

#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/program/arch.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/math/arithmetic.h"

namespace taichi {
namespace lang {
LlvmProgramImpl::LlvmProgramImpl(CompileConfig &config_) {
  runtime_mem_info = Runtime::create(config_.arch);
  if (config_.arch == Arch::cuda) {
    if (!runtime_mem_info) {
      TI_WARN("Taichi is not compiled with CUDA.");
      config_.arch = host_arch();
    } else if (!is_cuda_api_available()) {
      TI_WARN("No CUDA driver API detected.");
      config_.arch = host_arch();
    } else if (!runtime_mem_info->detected()) {
      TI_WARN("No CUDA device detected.");
      config_.arch = host_arch();
    } else {
      // CUDA runtime created successfully
    }
    if (config_.arch != Arch::cuda) {
      TI_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  }
  config = config_;
  snode_tree_buffer_manager = std::make_unique<SNodeTreeBufferManager>(this);

  thread_pool = std::make_unique<ThreadPool>(config.cpu_max_num_threads);

  preallocated_device_buffer = nullptr;
  llvm_context_host = std::make_unique<TaichiLLVMContext>(host_arch());
}

void LlvmProgramImpl::maybe_initialize_cuda_llvm_context() {
  if (config.arch == Arch::cuda && llvm_context_device == nullptr) {
    llvm_context_device = std::make_unique<TaichiLLVMContext>(Arch::cuda);
    llvm_context_device->init_runtime_jit_module();
  }
}

void LlvmProgramImpl::device_synchronize() {
  if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    TI_ERROR("No CUDA support");
#endif
  }
}

std::unique_ptr<llvm::Module>
LlvmProgramImpl::clone_struct_compiler_initial_context(
    const std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    TaichiLLVMContext *tlctx) {
  if (!snode_trees_.empty())
    return tlctx->clone_struct_module();
  return tlctx->clone_runtime_module();
}

void LlvmProgramImpl::initialize_llvm_runtime_snodes(const SNodeTree *tree,
                                                     StructCompiler *scomp,
                                                     uint64 *result_buffer) {
  TaichiLLVMContext *tlctx = nullptr;
  if (config.is_cuda_no_unified_memory()) {
#if defined(TI_WITH_CUDA)
    tlctx = llvm_context_device.get();
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    tlctx = llvm_context_host.get();
  }
  auto *const runtime_jit = tlctx->runtime_jit_module;
  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  const auto snodes = scomp->snodes;
  const int root_id = tree->root()->id;

  TI_TRACE("Allocating data structure of size {} bytes", scomp->root_size);
  std::size_t rounded_size =
      taichi::iroundup(scomp->root_size, taichi_page_size);
  runtime_jit->call<void *, std::size_t, int, int, int, std::size_t, Ptr>(
      "runtime_initialize_snodes", llvm_runtime, scomp->root_size, root_id,
      (int)snodes.size(), tree->id(), rounded_size,
      snode_tree_buffer_manager->allocate(runtime_jit, llvm_runtime,
                                          rounded_size, taichi_page_size,
                                          tree->id(), result_buffer));
  for (int i = 0; i < (int)snodes.size(); i++) {
    if (is_gc_able(snodes[i]->type)) {
      std::size_t node_size;
      auto element_size = snodes[i]->cell_size_bytes;
      if (snodes[i]->type == SNodeType::pointer) {
        // pointer. Allocators are for single elements
        node_size = element_size;
      } else {
        // dynamic. Allocators are for the chunks
        node_size = sizeof(void *) + element_size * snodes[i]->chunk_size;
      }
      TI_TRACE("Initializing allocator for snode {} (node size {})",
               snodes[i]->id, node_size);
      auto rt = llvm_runtime;
      runtime_jit->call<void *, int, std::size_t>(
          "runtime_NodeAllocator_initialize", rt, snodes[i]->id, node_size);
      TI_TRACE("Allocating ambient element for snode {} (node size {})",
               snodes[i]->id, node_size);
      runtime_jit->call<void *, int>("runtime_allocate_ambient", rt, i,
                                     node_size);
    }
  }
}

uint64 LlvmProgramImpl::fetch_result_uint64(int i, uint64 *result_buffer) {
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  device_synchronize();
  uint64 ret;
  auto arch = config.arch;
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    if (config.use_unified_memory) {
      // More efficient than a cudaMemcpy call in practice
      ret = result_buffer[i];
    } else {
      CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                       sizeof(uint64));
    }
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
  TI_ASSERT(arch_uses_llvm(config.arch));

  auto node_allocator =
      runtime_query<void *>("LLVMRuntime_get_node_allocators", result_buffer,
                            llvm_runtime, snode->id);
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

void LlvmProgramImpl::print_memory_profiler_info(
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config.arch));

  fmt::print("\n[Memory Profiler]\n");

  std::locale::global(std::locale("en_US.UTF-8"));
  // So that thousand separators are added to "{:n}" slots in fmtlib.
  // E.g., 10000 is printed as "10,000".
  // TODO: is there a way to set locale only locally in this function?

  std::function<void(SNode *, int)> visit = [&](SNode *snode, int depth) {
    auto element_list =
        runtime_query<void *>("LLVMRuntime_get_element_lists", result_buffer,
                              llvm_runtime, snode->id);

    if (snode->type != SNodeType::place) {
      fmt::print("SNode {:10}\n", snode->get_node_type_name_hinted());

      if (element_list) {
        fmt::print("  active element list:");
        print_list_manager_info(element_list, result_buffer);

        auto node_allocator =
            runtime_query<void *>("LLVMRuntime_get_node_allocators",
                                  result_buffer, llvm_runtime, snode->id);

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
      "LLVMRuntime_get_total_requested_memory", result_buffer, llvm_runtime);

  fmt::print(
      "Total requested dynamic memory (excluding alignment padding): {:n} B\n",
      total_requested_memory);
}
}  // namespace lang
}  // namespace taichi
