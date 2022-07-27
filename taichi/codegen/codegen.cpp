// Driver class for kernel codegen

#include "codegen.h"

#include "taichi/util/statistics.h"
#if defined(TI_WITH_LLVM)
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/codegen/wasm/codegen_wasm.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif
#if defined(TI_WITH_CUDA)
#include "taichi/codegen/cuda/codegen_cuda.h"
#endif
#include "taichi/system/timer.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

KernelCodeGen::KernelCodeGen(Kernel *kernel, IRNode *ir)
    : prog(kernel->program), kernel(kernel), ir(ir) {
  if (ir == nullptr)
    this->ir = kernel->ir.get();

  auto num_stmts = irpass::analysis::count_statements(this->ir);
  if (kernel->is_evaluator)
    stat.add("codegen_evaluator_statements", num_stmts);
  else if (kernel->is_accessor)
    stat.add("codegen_accessor_statements", num_stmts);
  else
    stat.add("codegen_kernel_statements", num_stmts);
  stat.add("codegen_statements", num_stmts);
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(Arch arch,
                                                     Kernel *kernel,
                                                     Stmt *stmt) {
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch) && arch != Arch::wasm) {
    return std::make_unique<KernelCodeGenCPU>(kernel, stmt);
  } else if (arch == Arch::wasm) {
    return std::make_unique<KernelCodeGenWASM>(kernel, stmt);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<KernelCodeGenCUDA>(kernel, stmt);
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

bool KernelCodeGen::maybe_read_compilation_from_cache(
    const std::string &kernel_key,
    std::vector<LLVMCompiledData> &data) {
  TI_AUTO_PROF;
  const auto &config = prog->config;
  auto reader =
      LlvmOfflineCacheFileReader::make(config.offline_cache_file_path);
  if (!reader) {
    return false;
  }

  LlvmOfflineCache::KernelCacheData cache_data;
  auto *tlctx = get_llvm_program(prog)->get_llvm_context(config.arch);
  auto &llvm_ctx = *tlctx->get_this_thread_context();

  if (!reader->get_kernel_cache(cache_data, kernel_key, llvm_ctx)) {
    return false;
  }
  data.swap(cache_data.compiled_data_list);
  kernel->set_from_offline_cache();
  return true;
}

void KernelCodeGen::cache_module(const std::string &kernel_key,
                                 const std::vector<LLVMCompiledData> &data) {
  get_llvm_program(prog)->cache_kernel(kernel_key, data,
                                       infer_launch_args(kernel));
}

ModuleToFunctionConverter::ModuleToFunctionConverter(
    TaichiLLVMContext *tlctx,
    LlvmRuntimeExecutor *executor)
    : tlctx_(tlctx), executor_(executor) {
}

FunctionType ModuleToFunctionConverter::convert(
    const Kernel *kernel,
    std::vector<LLVMCompiledData> &&data) const {
  return convert(kernel->name, infer_launch_args(kernel), std::move(data));
}

#endif
TLANG_NAMESPACE_END
