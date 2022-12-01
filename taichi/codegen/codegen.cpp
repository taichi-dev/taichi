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
#include "taichi/system/timer.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/analysis/offline_cache_util.h"

namespace taichi::lang {

KernelCodeGen::KernelCodeGen(Kernel *kernel, IRNode *ir)
    : prog(kernel->program), kernel(kernel), ir(ir) {
  if (ir == nullptr)
    this->ir = kernel->ir.get();
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
  } else if (arch == Arch::dx12) {
#if defined(TI_WITH_DX12)
    return std::make_unique<KernelCodeGenDX12>(kernel, stmt);
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
  const auto &config = prog->this_thread_config();
  auto *llvm_prog = get_llvm_program(prog);
  const auto &reader = llvm_prog->get_cache_reader();
  if (!reader) {
    return std::nullopt;
  }

  LlvmOfflineCache::KernelCacheData cache_data;
  auto *tlctx = llvm_prog->get_llvm_context(config.arch);
  auto &llvm_ctx = *tlctx->get_this_thread_context();

  if (!reader->get_kernel_cache(cache_data, kernel_key, llvm_ctx)) {
    return std::nullopt;
  }
  kernel->mark_as_from_cache();
  return {std::move(cache_data.compiled_data)};
}

void KernelCodeGen::cache_kernel(const std::string &kernel_key,
                                 const LLVMCompiledKernel &data) {
  get_llvm_program(prog)->cache_kernel(kernel_key, data,
                                       infer_launch_args(kernel));
}

LLVMCompiledKernel KernelCodeGen::compile_kernel_to_module() {
  auto *llvm_prog = get_llvm_program(prog);
  const auto &config = prog->this_thread_config();
  auto *tlctx = llvm_prog->get_llvm_context(config.arch);
  std::string kernel_key = get_hashed_offline_cache_key(&config, kernel);
  kernel->set_kernel_key_for_cache(kernel_key);
  if (config.offline_cache && this->supports_offline_cache() &&
      !kernel->is_evaluator) {
    auto res = maybe_read_compilation_from_cache(kernel_key);
    if (res) {
      TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
               kernel_key);
      cache_kernel(kernel_key, *res);
      return std::move(*res);
    }
  }
  if (!kernel->lowered()) {
    kernel->lower(/*to_executable=*/false);
  }

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  TI_ASSERT(block);

  auto &offloads = block->statements;
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(offloads.size());
  using TaskFunc = int32 (*)(void *);
  std::vector<TaskFunc> task_funcs(offloads.size());
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      tlctx->fetch_this_thread_struct_module();
      auto offload =
          irpass::analysis::clone(offloads[i].get(), offloads[i]->get_kernel());
      irpass::re_id(offload.get());
      auto new_data = this->compile_task(nullptr, offload->as<OffloadedStmt>());
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
  auto linked = tlctx->link_compiled_tasks(std::move(data));

  if (!kernel->is_evaluator) {
    TI_DEBUG("Cache kernel '{}' (key='{}')", kernel->get_name(), kernel_key);
    cache_kernel(kernel_key, linked);
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
  return convert(kernel->name, infer_launch_args(kernel), std::move(data));
}

#endif
}  // namespace taichi::lang
