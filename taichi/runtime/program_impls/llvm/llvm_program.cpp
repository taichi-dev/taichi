#include "taichi/runtime/program_impls/llvm/llvm_program.h"

#include "llvm/IR/Module.h"

#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/program/program.h"
#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/llvm/llvm_aot_module_builder.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/analysis/offline_cache_util.h"

#if defined(TI_WITH_CUDA)
#include "taichi/codegen/cuda/codegen_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

#if defined(TI_WITH_AMDGPU)
#include "taichi/codegen/amdgpu/codegen_amdgpu.h"
#include "taichi/runtime/amdgpu/kernel_launcher.h"
#endif

#if defined(TI_WITH_DX12)
#include "taichi/runtime/dx12/aot_module_builder_impl.h"
#include "taichi/codegen/dx12/codegen_dx12.h"
#endif

#include "taichi/codegen/llvm/kernel_compiler.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"

namespace taichi::lang {
LlvmProgramImpl::LlvmProgramImpl(CompileConfig &config_,
                                 KernelProfilerBase *profiler)
    : ProgramImpl(config_),
      compilation_workers("compile",
                          config_.print_ir ? 1 : config_.num_compile_threads) {
  runtime_exec_ = std::make_unique<LlvmRuntimeExecutor>(config_, profiler);
  cache_data_ = std::make_unique<LlvmOfflineCache>();
}

std::unique_ptr<StructCompiler> LlvmProgramImpl::compile_snode_tree_types_impl(
    SNodeTree *tree) {
  auto *const root = tree->root();
  std::unique_ptr<StructCompiler> struct_compiler{nullptr};
  auto module = runtime_exec_->llvm_context_.get()->new_module("struct");
  struct_compiler = std::make_unique<StructCompilerLLVM>(
      arch_is_cpu(config->arch) ? host_arch() : config->arch, this,
      std::move(module), tree->id());
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

  TI_ASSERT(cache_data_->fields.find(snode_tree_id) !=
            cache_data_->fields.end());
  initialize_llvm_runtime_snodes(cache_data_->fields.at(snode_tree_id),
                                 result_buffer);
}

std::unique_ptr<AotModuleBuilder> LlvmProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (config->arch == Arch::x64 || config->arch == Arch::arm64 ||
      config->arch == Arch::cuda) {
    return std::make_unique<LlvmAotModuleBuilder>(
        get_kernel_compilation_manager(), *config, this);
  }

#if defined(TI_WITH_DX12)
  if (config->arch == Arch::dx12) {
    return std::make_unique<directx12::AotModuleBuilderImpl>(
        *config, this, *runtime_exec_->get_llvm_context());
  }
#endif

  TI_NOT_IMPLEMENTED;
  return nullptr;
}

void LlvmProgramImpl::cache_field(int snode_tree_id,
                                  int root_id,
                                  const StructCompiler &struct_compiler) {
  if (cache_data_->fields.find(snode_tree_id) != cache_data_->fields.end()) {
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

  cache_data_->fields[snode_tree_id] = std::move(ret);
}

std::unique_ptr<KernelCompiler> LlvmProgramImpl::make_kernel_compiler() {
  lang::LLVM::KernelCompiler::Config cfg;
  cfg.tlctx = runtime_exec_->get_llvm_context();
  return std::make_unique<lang::LLVM::KernelCompiler>(std::move(cfg));
}

std::unique_ptr<KernelLauncher> LlvmProgramImpl::make_kernel_launcher() {
  LLVM::KernelLauncher::Config cfg;
  cfg.executor = runtime_exec_.get();

  if (arch_is_cpu(config->arch)) {
    return std::make_unique<cpu::KernelLauncher>(std::move(cfg));
  } else if (config->arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<cuda::KernelLauncher>(std::move(cfg));
#endif
  } else if (config->arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    return std::make_unique<amdgpu::KernelLauncher>(std::move(cfg));
#endif
  } else if (config->arch == Arch::dx12) {
    // Not implemented
  }

  TI_NOT_IMPLEMENTED;
}

LlvmProgramImpl *get_llvm_program(Program *prog) {
  LlvmProgramImpl *llvm_prog =
      dynamic_cast<LlvmProgramImpl *>(prog->get_program_impl());
  TI_ASSERT(llvm_prog != nullptr);
  return llvm_prog;
}

}  // namespace taichi::lang
