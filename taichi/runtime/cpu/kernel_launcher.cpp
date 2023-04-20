#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {
namespace cpu {

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  TI_ASSERT(handle.get_launch_id() < contexts_.size());
  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();

  ctx.get_context().runtime = executor->get_llvm_runtime();
  // For taichi ndarrays, context.array_ptrs saves pointer to its
  // |DeviceAllocation|, CPU backend actually want to use the raw ptr here.
  const auto &parameters = launcher_ctx.parameters;
  for (int i = 0; i < (int)parameters.size(); i++) {
    if (parameters[i].is_array &&
        ctx.device_allocation_type[i] ==
            LaunchContextBuilder::DevAllocType::kNone) {
      ctx.set_struct_arg({i, 1}, (uint64)ctx.array_ptrs[{i}]);
    }
    if (parameters[i].is_array &&
        ctx.device_allocation_type[i] !=
            LaunchContextBuilder::DevAllocType::kNone &&
        ctx.array_runtime_sizes[i] > 0) {
      DeviceAllocation *ptr =
          static_cast<DeviceAllocation *>(ctx.array_ptrs[{i}]);
      uint64 host_ptr = (uint64)executor->get_ndarray_alloc_info_ptr(*ptr);
      ctx.set_struct_arg({i, 1}, host_ptr);
      ctx.set_array_device_allocation_type(
          i, LaunchContextBuilder::DevAllocType::kNone);

      if (ctx.has_grad[i]) {
        DeviceAllocation *ptr_grad =
            static_cast<DeviceAllocation *>(ctx.get_grad_arg<void *>(i));
        uint64 host_ptr_grad =
            (uint64)executor->get_ndarray_alloc_info_ptr(*ptr_grad);
        ctx.set_grad_arg(i, host_ptr_grad);
      }
    }
  }
  for (auto task : launcher_ctx.task_funcs) {
    task(&ctx.get_context());
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(arch_is_cpu(compiled.arch()));

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto parameters = compiled.get_internal_data().args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Construct task_funcs
    using TaskFunc = int32 (*)(void *);
    std::vector<TaskFunc> task_funcs;
    task_funcs.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      TI_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found",
                     task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
    }

    // Populate ctx
    ctx.parameters = std::move(parameters);
    ctx.task_funcs = std::move(task_funcs);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cpu
}  // namespace taichi::lang
