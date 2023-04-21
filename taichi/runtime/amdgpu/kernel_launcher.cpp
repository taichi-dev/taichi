#include "taichi/runtime/amdgpu/kernel_launcher.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"

namespace taichi::lang {
namespace amdgpu {

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  TI_ASSERT(handle.get_launch_id() < contexts_.size());
  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *amdgpu_module = launcher_ctx.jit_module;
  const auto &parameters = launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  AMDGPUContext::get_instance().make_current();
  ctx.get_context().runtime = executor->get_llvm_runtime();
  std::vector<void *> arg_buffers(parameters.size(), nullptr);
  std::vector<void *> device_buffers(parameters.size(), nullptr);
  char *device_result_buffer{nullptr};
  bool transferred = false;
  for (int i = 0; i < (int)parameters.size(); i++) {
    if (parameters[i].is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[i];
      if (arr_sz == 0)
        continue;
      arg_buffers[i] =
          ctx.array_ptrs[{i, TypeFactory::DATA_PTR_POS_IN_NDARRAY}];
      if (ctx.device_allocation_type[i] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        unsigned int attr_val[8];
        uint32_t ret_code =
            AMDGPUDriver::get_instance().mem_get_attributes.call(
                attr_val, (void *)arg_buffers[i]);
        if (ret_code != HIP_SUCCESS || attr_val[0] != HIP_MEMORYTYPE_DEVICE) {
          transferred = true;
          AMDGPUDriver::get_instance().malloc(&device_buffers[i], arr_sz);
          AMDGPUDriver::get_instance().memcpy_host_to_device(
              (void *)device_buffers[i], arg_buffers[i], arr_sz);
        } else {
          device_buffers[i] = arg_buffers[i];
        }
        ctx.set_ndarray_ptrs(
            i, (uint64)device_buffers[i],
            (uint64)ctx.array_ptrs[{i, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}]);

      } else if (arr_sz > 0) {  // why use arr_sz constrain?
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(arg_buffers[i]);
        device_buffers[i] = executor->get_ndarray_alloc_info_ptr(*ptr);
        arg_buffers[i] = device_buffers[i];
        ctx.set_ndarray_ptrs(
            i, (uint64)device_buffers[i],
            (uint64)ctx.array_ptrs[{i, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}]);
      }
    }
  }
  if (transferred) {
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc((void **)&device_result_buffer,
                                        ctx.result_buffer_size);
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc((void **)&device_arg_buffer,
                                        ctx.arg_buffer_size);
    AMDGPUDriver::get_instance().memcpy_host_to_device(
        device_arg_buffer, ctx.get_context().arg_buffer, ctx.arg_buffer_size);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }
  void *context_pointer;
  int arg_size = sizeof(RuntimeContext *);
  AMDGPUDriver::get_instance().malloc((void **)&context_pointer,
                                      sizeof(RuntimeContext));
  AMDGPUDriver::get_instance().memcpy_host_to_device(
      context_pointer, &ctx.get_context(), sizeof(RuntimeContext));

  AMDGPUContext::get_instance().push_back_kernel_arg_pointer(context_pointer);

  for (auto &task : offloaded_tasks) {
    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    amdgpu_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                          {(void *)&context_pointer}, {arg_size});
  }
  TI_TRACE("Launching kernel");
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().mem_free(device_arg_buffer);
  }
  if (ctx.result_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_device_to_host(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size);
    AMDGPUDriver::get_instance().mem_free(device_result_buffer);
  }
  if (transferred) {
    for (int i = 0; i < parameters.size(); i++) {
      if (device_buffers[i] != arg_buffers[i]) {
        AMDGPUDriver::get_instance().memcpy_device_to_host(
            arg_buffers[i], (void *)device_buffers[i],
            ctx.array_runtime_sizes[i]);
        AMDGPUDriver::get_instance().mem_free((void *)device_buffers[i]);
      }
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::amdgpu);

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto parameters = compiled.get_internal_data().args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx.jit_module = jit_module;
    ctx.parameters = std::move(parameters);
    ctx.offloaded_tasks = std::move(data.tasks);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace amdgpu
}  // namespace taichi::lang
