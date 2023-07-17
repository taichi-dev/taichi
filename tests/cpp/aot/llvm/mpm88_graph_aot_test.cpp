#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
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

constexpr int NR_PARTICLES = 8192 * 5;
constexpr int N_GRID = 128;

TEST(LlvmCGraph, Mpm88Cpu) {
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

  // Prepare & Run "init" Graph
  auto g_init = mod->get_graph("init");

  /* Prepare arguments */
  constexpr int kArrBytes_x = NR_PARTICLES * 2 * sizeof(float);
  auto devalloc_x = exec.allocate_memory_on_device(kArrBytes_x, result_buffer);
  auto x = taichi::lang::Ndarray(devalloc_x, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES}, {2});

  constexpr int kArrBytes_v = NR_PARTICLES * 2 * sizeof(float);
  auto devalloc_v = exec.allocate_memory_on_device(kArrBytes_v, result_buffer);
  auto v = taichi::lang::Ndarray(devalloc_v, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES}, {2});

  constexpr int kArrBytes_J = NR_PARTICLES * sizeof(float);
  auto devalloc_J = exec.allocate_memory_on_device(kArrBytes_J, result_buffer);
  auto J = taichi::lang::Ndarray(devalloc_J, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES});

  std::unordered_map<std::string, taichi::lang::aot::IValue> args;
  args.insert({"x", taichi::lang::aot::IValue::create(x)});
  args.insert({"v", taichi::lang::aot::IValue::create(v)});
  args.insert({"J", taichi::lang::aot::IValue::create(J)});

  g_init->run(args);
  exec.synchronize();

  // Prepare & Run "update" Graph
  auto g_update = mod->get_graph("update");

  constexpr int kArrBytes_grid_v = N_GRID * N_GRID * 2 * sizeof(float);
  auto devalloc_grid_v =
      exec.allocate_memory_on_device(kArrBytes_grid_v, result_buffer);
  auto grid_v = taichi::lang::Ndarray(
      devalloc_grid_v, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID}, {2});

  constexpr int kArrBytes_grid_m = N_GRID * N_GRID * sizeof(float);
  auto devalloc_grid_m =
      exec.allocate_memory_on_device(kArrBytes_grid_m, result_buffer);
  auto grid_m = taichi::lang::Ndarray(
      devalloc_grid_m, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID});

  constexpr int kArrBytes_pos = NR_PARTICLES * 3 * sizeof(float);
  auto devalloc_pos =
      exec.allocate_memory_on_device(kArrBytes_pos, result_buffer);
  auto pos = taichi::lang::Ndarray(
      devalloc_pos, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});

  constexpr int kArrBytes_C = NR_PARTICLES * sizeof(float) * 2 * 2;
  auto devalloc_C = exec.allocate_memory_on_device(kArrBytes_C, result_buffer);
  auto C = taichi::lang::Ndarray(devalloc_C, taichi::lang::PrimitiveType::f32,
                                 {NR_PARTICLES}, {2, 2});

  args.insert({"C", taichi::lang::aot::IValue::create(C)});
  args.insert({"grid_v", taichi::lang::aot::IValue::create(grid_v)});
  args.insert({"grid_m", taichi::lang::aot::IValue::create(grid_m)});
  args.insert({"pos", taichi::lang::aot::IValue::create(pos)});

  g_update->run(args);
  exec.synchronize();
}

TEST(LlvmCGraph, Mpm88Cuda) {
#ifdef TI_WITH_CUDA
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmRuntimeExecutor exec{cfg, kNoProfiler};
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

    // Prepare & Run "init" Graph
    auto g_init = mod->get_graph("init");

    /* Prepare arguments */
    constexpr int kArrBytes_x = NR_PARTICLES * 2 * sizeof(float);
    auto devalloc_x =
        exec.allocate_memory_on_device(kArrBytes_x, result_buffer);
    auto x = taichi::lang::Ndarray(devalloc_x, taichi::lang::PrimitiveType::f32,
                                   {NR_PARTICLES}, {2});

    constexpr int kArrBytes_v = NR_PARTICLES * 2 * sizeof(float);
    auto devalloc_v =
        exec.allocate_memory_on_device(kArrBytes_v, result_buffer);
    auto v = taichi::lang::Ndarray(devalloc_v, taichi::lang::PrimitiveType::f32,
                                   {NR_PARTICLES}, {2});

    constexpr int kArrBytes_J = NR_PARTICLES * sizeof(float);
    auto devalloc_J =
        exec.allocate_memory_on_device(kArrBytes_J, result_buffer);
    auto J = taichi::lang::Ndarray(devalloc_J, taichi::lang::PrimitiveType::f32,
                                   {NR_PARTICLES});

    std::unordered_map<std::string, taichi::lang::aot::IValue> args;
    args.insert({"x", taichi::lang::aot::IValue::create(x)});
    args.insert({"v", taichi::lang::aot::IValue::create(v)});
    args.insert({"J", taichi::lang::aot::IValue::create(J)});

    g_init->run(args);
    exec.synchronize();

    // Prepare & Run "update" Graph
    auto g_update = mod->get_graph("update");

    constexpr int kArrBytes_grid_v = N_GRID * N_GRID * 2 * sizeof(float);
    auto devalloc_grid_v =
        exec.allocate_memory_on_device(kArrBytes_grid_v, result_buffer);
    auto grid_v =
        taichi::lang::Ndarray(devalloc_grid_v, taichi::lang::PrimitiveType::f32,
                              {N_GRID, N_GRID}, {2});

    constexpr int kArrBytes_grid_m = N_GRID * N_GRID * sizeof(float);
    auto devalloc_grid_m =
        exec.allocate_memory_on_device(kArrBytes_grid_m, result_buffer);
    auto grid_m = taichi::lang::Ndarray(
        devalloc_grid_m, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID});

    constexpr int kArrBytes_pos = NR_PARTICLES * 3 * sizeof(float);
    auto devalloc_pos =
        exec.allocate_memory_on_device(kArrBytes_pos, result_buffer);
    auto pos = taichi::lang::Ndarray(
        devalloc_pos, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});

    constexpr int kArrBytes_C = NR_PARTICLES * sizeof(float) * 2 * 2;
    auto devalloc_C =
        exec.allocate_memory_on_device(kArrBytes_C, result_buffer);
    auto C = taichi::lang::Ndarray(devalloc_C, taichi::lang::PrimitiveType::f32,
                                   {NR_PARTICLES}, {2, 2});

    args.insert({"C", taichi::lang::aot::IValue::create(C)});
    args.insert({"grid_v", taichi::lang::aot::IValue::create(grid_v)});
    args.insert({"grid_m", taichi::lang::aot::IValue::create(grid_m)});
    args.insert({"pos", taichi::lang::aot::IValue::create(pos)});

    g_update->run(args);
    exec.synchronize();
  }
#endif
}
