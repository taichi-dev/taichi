#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/system/memory_pool.h"
#include "taichi/runtime/cpu/aot_module_loader_impl.h"
#include "taichi/runtime/cuda/aot_module_loader_impl.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {

TEST(LlvmAotTest, CpuKernel) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmRuntimeExecutor exec{cfg, kNoProfiler};
  auto *compute_device = exec.get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  auto memory_pool = std::make_unique<MemoryPool>(cfg.arch, compute_device);
  uint64 *result_buffer{nullptr};
  exec.materialize_runtime(memory_pool.get(), kNoProfiler, &result_buffer);

  constexpr int kArrLen = 32;
  constexpr int kArrBytes = kArrLen * sizeof(int32_t);
  auto arr_devalloc = exec.allocate_memory_ndarray(kArrBytes, result_buffer);
  Ndarray arr = Ndarray(arr_devalloc, PrimitiveType::i32, {kArrLen});

  cpu::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.executor_ = &exec;
  auto mod = cpu::make_aot_module(aot_params);
  auto *k_run = mod->get_kernel("run");

  RuntimeContext ctx;
  ctx.runtime = exec.get_llvm_runtime();
  ctx.set_arg(0, /*v=*/0);
  ctx.set_arg_ndarray(/*arg_id=*/1, arr.get_device_allocation_ptr_as_int(),
                      /*shape=*/arr.shape);
  k_run->launch(&ctx);

  auto *data = reinterpret_cast<int32_t *>(
      exec.get_ndarray_alloc_info_ptr(arr_devalloc));
  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], i);
  }
}

TEST(LlvmAotTest, CudaKernel) {
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmRuntimeExecutor exec{cfg, kNoProfiler};

    // Must have handled all the arch fallback logic by this point.
    uint64 *result_buffer{nullptr};
    exec.materialize_runtime(nullptr, kNoProfiler, &result_buffer);

    constexpr int kArrLen = 32;
    constexpr int kArrBytes = kArrLen * sizeof(int32_t);
    auto arr_devalloc = exec.allocate_memory_ndarray(kArrBytes, result_buffer);
    Ndarray arr = Ndarray(arr_devalloc, PrimitiveType::i32, {kArrLen});

    cuda::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.executor_ = &exec;
    auto mod = cuda::make_aot_module(aot_params);
    auto *k_run = mod->get_kernel("run");
    RuntimeContext ctx;
    ctx.runtime = exec.get_llvm_runtime();
    ctx.set_arg(0, /*v=*/0);
    ctx.set_arg_ndarray(/*arg_id=*/1, arr.get_device_allocation_ptr_as_int(),
                        /*shape=*/arr.shape);
    k_run->launch(&ctx);

    auto *data = reinterpret_cast<int32_t *>(
        exec.get_ndarray_alloc_info_ptr(arr_devalloc));

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
