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
#include "taichi/ir/transforms.h"
#include "taichi/analysis/offline_cache_util.h"

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
  auto *llvm_prog = get_llvm_program(prog);
  const auto &reader = llvm_prog->get_cache_reader();
  if (!reader) {
    return false;
  }

  LlvmOfflineCache::KernelCacheData cache_data;
  auto *tlctx = llvm_prog->get_llvm_context(config.arch);
  auto &llvm_ctx = *tlctx->get_this_thread_context();

  if (!reader->get_kernel_cache(cache_data, kernel_key, llvm_ctx)) {
    return false;
  }
  data.swap(cache_data.compiled_data_list);
  kernel->mark_as_from_cache();
  return true;
}

void KernelCodeGen::cache_module(const std::string &kernel_key,
                                 const std::vector<LLVMCompiledData> &data) {
  get_llvm_program(prog)->cache_kernel(kernel_key, data,
                                       infer_launch_args(kernel));
}

std::vector<LLVMCompiledData> KernelCodeGen::compile_kernel_to_module() {
  auto &config = prog->config;
  std::string kernel_key = get_hashed_offline_cache_key(&config, kernel);
  kernel->set_kernel_key_for_cache(kernel_key);
  if (config.offline_cache && this->supports_offline_cache() &&
      !kernel->is_evaluator) {
    std::vector<LLVMCompiledData> res;
    const bool ok = maybe_read_compilation_from_cache(kernel_key, res);
    if (ok) {
      TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
               kernel_key);
      cache_module(kernel_key, res);
      return res;
    }
  }
  if (!kernel->lowered()) {
    kernel->lower(/*to_executable=*/false);
  }

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  TI_ASSERT(block);

  auto &offloads = block->statements;
  std::vector<LLVMCompiledData> data(offloads.size());
  using TaskFunc = int32 (*)(void *);
  std::vector<TaskFunc> task_funcs(offloads.size());
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      auto offload =
          irpass::analysis::clone(offloads[i].get(), offloads[i]->get_kernel());
      irpass::re_id(offload.get());
      auto new_data = this->compile_task(nullptr, offload->as<OffloadedStmt>());
      data[i].tasks = std::move(new_data.tasks);
      data[i].module = std::move(new_data.module);
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
  if (!kernel->is_evaluator) {
    TI_DEBUG("Cache kernel '{}' (key='{}')", kernel->get_name(), kernel_key);
    cache_module(kernel_key, data);
  }
  return data;
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
