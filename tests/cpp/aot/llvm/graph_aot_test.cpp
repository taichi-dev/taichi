#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/cpu/kernel_launcher.h"

#ifdef TI_WITH_CUDA

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

#endif

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

using namespace taichi;
using namespace lang;

TEST(LlvmCGraph, RunGraphCpu) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmRuntimeExecutor exec{cfg, kNoProfiler};
  // Must have handled all the arch fallback logic by this point.
  uint64 *result_buffer{nullptr};
  exec.materialize_runtime(kNoProfiler, &result_buffer);

  /* AOTLoader */
  LLVM::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.executor_ = &exec;
  aot_params.kernel_launcher =
      std::make_unique<cpu::KernelLauncher>(cpu::KernelLauncher::Config{&exec});
  std::unique_ptr<aot::Module> mod =
      LLVM::make_aot_module(std::move(aot_params));

  constexpr int ArrLength = 100;
  constexpr int kArrBytes_arr = ArrLength * 1 * sizeof(int32_t);
  auto devalloc_arr_0 =
      exec.allocate_memory_on_device(kArrBytes_arr, result_buffer);
  auto devalloc_arr_1 =
      exec.allocate_memory_on_device(kArrBytes_arr, result_buffer);

  /* Test with Graph */
  // Prepare & Run "init" Graph
  auto run_graph = mod->get_graph("run_graph");

  auto arr0 = taichi::lang::Ndarray(
      devalloc_arr_0, taichi::lang::PrimitiveType::i32, {ArrLength});
  auto arr1 = taichi::lang::Ndarray(
      devalloc_arr_1, taichi::lang::PrimitiveType::i32, {ArrLength},
      {
          1,
      });

  int base0 = 10;
  int base1 = 20;
  int base2 = 30;
  std::unordered_map<std::string, taichi::lang::aot::IValue> args;
  args.insert({"arr0", taichi::lang::aot::IValue::create(arr0)});
  args.insert({"arr1", taichi::lang::aot::IValue::create(arr1)});
  args.insert({"base0", taichi::lang::aot::IValue::create(base0)});
  args.insert({"base1", taichi::lang::aot::IValue::create(base1)});
  args.insert({"base2", taichi::lang::aot::IValue::create(base2)});

  run_graph->run(args);
  exec.synchronize();

  auto *data_0 = reinterpret_cast<int32_t *>(
      exec.get_device_alloc_info_ptr(devalloc_arr_0));
  auto *data_1 = reinterpret_cast<int32_t *>(
      exec.get_device_alloc_info_ptr(devalloc_arr_1));
  for (int i = 0; i < ArrLength; i++) {
    EXPECT_EQ(data_0[i], 3 * i + base0 + base1 + base2);
  }
  for (int i = 0; i < ArrLength; i++) {
    EXPECT_EQ(data_1[i], 3 * i + base0 + base1 + base2);
  }
}

TEST(LlvmCGraph, RunGraphCuda) {
#ifdef TI_WITH_CUDA
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmRuntimeExecutor exec{cfg, kNoProfiler};

    // Must have handled all the arch fallback logic by this point.
    uint64 *result_buffer{nullptr};
    exec.materialize_runtime(kNoProfiler, &result_buffer);

    /* AOTLoader */
    LLVM::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.executor_ = &exec;
    aot_params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
        cuda::KernelLauncher::Config{&exec});
    auto mod = LLVM::make_aot_module(std::move(aot_params));

    constexpr int ArrLength = 100;
    constexpr int kArrBytes_arr = ArrLength * 1 * sizeof(int32_t);
    auto devalloc_arr_0 =
        exec.allocate_memory_on_device(kArrBytes_arr, result_buffer);

    auto devalloc_arr_1 =
        exec.allocate_memory_on_device(kArrBytes_arr, result_buffer);

    /* Test with Graph */
    // Prepare & Run "init" Graph
    auto run_graph = mod->get_graph("run_graph");

    auto arr0 = taichi::lang::Ndarray(
        devalloc_arr_0, taichi::lang::PrimitiveType::i32, {ArrLength});

    auto arr1 = taichi::lang::Ndarray(
        devalloc_arr_1, taichi::lang::PrimitiveType::i32, {ArrLength}, {1});

    int base0 = 10;
    int base1 = 20;
    int base2 = 30;
    std::unordered_map<std::string, taichi::lang::aot::IValue> args;
    args.insert({"arr0", taichi::lang::aot::IValue::create(arr0)});
    args.insert({"arr1", taichi::lang::aot::IValue::create(arr1)});
    args.insert({"base0", taichi::lang::aot::IValue::create(base0)});
    args.insert({"base1", taichi::lang::aot::IValue::create(base1)});
    args.insert({"base2", taichi::lang::aot::IValue::create(base2)});

    run_graph->run(args);
    exec.synchronize();

    std::vector<int32_t> cpu_data(ArrLength);

    auto *data_0 = reinterpret_cast<int32_t *>(
        exec.get_device_alloc_info_ptr(devalloc_arr_0));

    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)cpu_data.data(), (void *)data_0, ArrLength * sizeof(int32_t));

    for (int i = 0; i < ArrLength; ++i) {
      EXPECT_EQ(cpu_data[i], 3 * i + base0 + base1 + base2);
    }

    auto *data_1 = reinterpret_cast<int32_t *>(
        exec.get_device_alloc_info_ptr(devalloc_arr_1));

    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)cpu_data.data(), (void *)data_1, ArrLength * sizeof(int32_t));

    for (int i = 0; i < ArrLength; ++i) {
      EXPECT_EQ(cpu_data[i], 3 * i + base0 + base1 + base2);
    }
  }
#endif
}
