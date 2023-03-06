// Driver class for kernel codegen

#include "codegen.h"

#if defined(TI_WITH_LLVM)
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/codegen/wasm/codegen_wasm.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif
#if defined(TI_WITH_CUDA)
#include "taichi/codegen/cuda/codegen_cuda.h"
#endif
#if defined(TI_WITH_DX12)
#include "taichi/codegen/dx12/codegen_dx12.h"
#endif
#if defined(TI_WITH_AMDGPU)
#include "taichi/codegen/amdgpu/codegen_amdgpu.h"
#endif
#include "taichi/system/timer.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/analysis/offline_cache_util.h"

namespace taichi::lang {

KernelCodeGen::KernelCodeGen(const CompileConfig &compile_config,
                             Kernel *kernel,
                             TaichiLLVMContext &tlctx)
    : prog(kernel->program),
      kernel(kernel),
      compile_config_(compile_config),
      tlctx_(tlctx) {
  this->ir = kernel->ir.get();
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(
    const CompileConfig &compile_config,
    Kernel *kernel,
    TaichiLLVMContext &tlctx) {
#ifdef TI_WITH_LLVM
  const auto arch = compile_config.arch;
  if (arch_is_cpu(arch) && arch != Arch::wasm) {
    return std::make_unique<KernelCodeGenCPU>(compile_config, kernel, tlctx);
  } else if (arch == Arch::wasm) {
    return std::make_unique<KernelCodeGenWASM>(compile_config, kernel, tlctx);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<KernelCodeGenCUDA>(compile_config, kernel, tlctx);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::dx12) {
#if defined(TI_WITH_DX12)
    return std::make_unique<KernelCodeGenDX12>(compile_config, kernel, tlctx);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    return std::make_unique<KernelCodeGenAMDGPU>(compile_config, kernel, tlctx);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }
#else
  TI_ERROR("Llvm disabled");
#endif
}
#ifdef TI_WITH_LLVM

std::optional<LLVMCompiledKernel>
KernelCodeGen::maybe_read_compilation_from_cache(
    const std::string &kernel_key) {
  TI_AUTO_PROF;
  auto *llvm_prog = get_llvm_program(prog);
  const auto &reader = llvm_prog->get_cache_reader();
  if (!reader) {
    return std::nullopt;
  }

  LlvmOfflineCache::KernelCacheData cache_data;
  auto &llvm_ctx = *tlctx_.get_this_thread_context();

  if (!reader->get_kernel_cache(cache_data, kernel_key, llvm_ctx)) {
    return std::nullopt;
  }
  return {std::move(cache_data.compiled_data)};
}

void KernelCodeGen::cache_kernel(const std::string &kernel_key,
                                 const LLVMCompiledKernel &data) {
  get_llvm_program(prog)->cache_kernel(kernel_key, data, kernel);
}

LLVMCompiledKernel KernelCodeGen::compile_kernel_to_module() {
  // NOTE: The code below (codegen + offline cache) is a little confusing. But
  // don't worry, the KernelCompilationManager to be introduced to unify the
  // implementation of the offline cache will resolve the problem.

  bool enable_offline_cache = compile_config_.offline_cache &&
                              this->supports_offline_cache() &&
                              !kernel->is_evaluator;

  // Generate offline cache key & Try loading kernel from offline cache
  if (enable_offline_cache) {
    std::string key = get_hashed_offline_cache_key(compile_config_, {}, kernel);
    kernel->set_kernel_key_for_cache(key);
    if (auto res = maybe_read_compilation_from_cache(key)) {
      TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
               key);
      return std::move(*res);
    }
  }

  // Compile & Cache it

  irpass::ast_to_ir(compile_config_, *kernel, false);

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  TI_ASSERT(block);

  auto &offloads = block->statements;
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(offloads.size());
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());
      auto new_data = this->compile_task(compile_config_, nullptr,
                                         offload->as<OffloadedStmt>());
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
    };
    if (kernel->is_evaluator) {
      compile_func();
    } else {
      worker.enqueue(compile_func);
    }
  }
  if (!kernel->is_evaluator) {
    worker.flush();
  }
  auto linked = tlctx_.link_compiled_tasks(std::move(data));

  if (enable_offline_cache) {
    const auto &key = kernel->get_cached_kernel_key();
    TI_ASSERT(!key.empty());
    TI_DEBUG("Cache kernel '{}' (key='{}')", kernel->get_name(), key);
    cache_kernel(key, linked);
  }

  return linked;
}

ModuleToFunctionConverter::ModuleToFunctionConverter(
    TaichiLLVMContext *tlctx,
    LlvmRuntimeExecutor *executor)
    : tlctx_(tlctx), executor_(executor) {
}

FunctionType ModuleToFunctionConverter::convert(const Kernel *kernel,
                                                LLVMCompiledKernel data) const {
  return convert(kernel->name, kernel->parameter_list, std::move(data));
}

#endif
}  // namespace taichi::lang
