#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/system/memory_pool.h"
#include "taichi/backends/cpu/aot_module_loader_impl.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {

TEST(LlvmProgramTest, FullPipeline) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmProgramImpl prog{cfg, kNoProfiler};
  auto *compute_device = prog.get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  auto memory_pool = std::make_unique<MemoryPool>(cfg.arch, compute_device);
  prog.initialize_host();
  uint64 *result_buffer{nullptr};
  prog.materialize_runtime(memory_pool.get(), kNoProfiler, &result_buffer);

  cpu::AotModuleParams aot_params;
  aot_params.module_path = "generated";
  aot_params.program = &prog;
  auto mod = cpu::make_aot_module(aot_params);
  auto *k_run = mod->get_kernel("run");
  RuntimeContext ctx;
  ctx.runtime = prog.get_llvm_runtime();
  k_run->launch(&ctx);
}

}  // namespace lang
}  // namespace taichi
