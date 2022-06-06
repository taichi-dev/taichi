#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/system/memory_pool.h"
#include "taichi/backends/cpu/aot_module_loader_impl.h"
#include "taichi/backends/cuda/aot_module_loader_impl.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"

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

  constexpr int kArrLen = 32;
  constexpr int kArrBytes = kArrLen * sizeof(int32_t);
  auto arr_devalloc = prog.allocate_memory_ndarray(kArrBytes, result_buffer);

  cpu::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.program = &prog;
  auto mod = cpu::make_aot_module(aot_params);
  auto *k_run = mod->get_kernel("run");
  RuntimeContext ctx;
  ctx.runtime = prog.get_llvm_runtime();
  ctx.set_arg(0, /*v=*/0);
  ctx.set_arg_devalloc(/*arg_id=*/1, arr_devalloc, /*shape=*/{kArrLen});
  ctx.set_array_runtime_size(/*arg_id=*/1, kArrBytes);
  k_run->launch(&ctx);

  auto *data = reinterpret_cast<int32_t *>(
      prog.get_ndarray_alloc_info_ptr(arr_devalloc));
  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], i);
  }
}

TEST(LlvmProgramTest, FullPipelineCUDA) {
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmProgramImpl prog{cfg, kNoProfiler};

    // Must have handled all the arch fallback logic by this point.
    prog.initialize_host();
    uint64 *result_buffer{nullptr};
    prog.materialize_runtime(nullptr, kNoProfiler, &result_buffer);

    constexpr int kArrLen = 32;
    constexpr int kArrBytes = kArrLen * sizeof(int32_t);
    auto arr_devalloc = prog.allocate_memory_ndarray(kArrBytes, result_buffer);

    cuda::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.program = &prog;
    auto mod = cuda::make_aot_module(aot_params);
    auto *k_run = mod->get_kernel("run");
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    ctx.set_arg(0, /*v=*/0);
    ctx.set_arg_devalloc(/*arg_id=*/1, arr_devalloc, /*shape=*/{kArrLen});
    ctx.set_array_runtime_size(/*arg_id=*/1, kArrBytes);
    k_run->launch(&ctx);

    auto *data = reinterpret_cast<int32_t *>(
        prog.get_ndarray_alloc_info_ptr(arr_devalloc));

    std::vector<int32_t> cpu_data(kArrLen);
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)cpu_data.data(), (void *)data, kArrLen * sizeof(int32_t));

    for (int i = 0; i < kArrLen; ++i) {
      EXPECT_EQ(cpu_data[i], i);
    }
  }
}

}  // namespace lang
}  // namespace taichi
