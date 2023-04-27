#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang {
namespace cuda {

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  TI_ASSERT(handle.get_launch_id() < contexts_.size());
  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *cuda_module = launcher_ctx.jit_module;
  const auto &parameters = launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  CUDAContext::get_instance().make_current();
  std::vector<void *> arg_buffers(parameters.size(), nullptr);
  std::vector<void *> device_buffers(parameters.size(), nullptr);
  std::vector<DeviceAllocation> temporary_devallocs(parameters.size());
  char *device_result_buffer{nullptr};
  CUDADriver::get_instance().malloc_async(
      (void **)&device_result_buffer,
      std::max(ctx.result_buffer_size, sizeof(uint64)), nullptr);
  ctx.get_context().runtime = executor->get_llvm_runtime();

  bool transferred = false;
  for (int i = 0; i < (int)parameters.size(); i++) {
    if (parameters[i].is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[i];
      if (arr_sz == 0) {
        continue;
      }
      arg_buffers[i] = ctx.array_ptrs[{i}];
      if (ctx.device_allocation_type[i] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        // Note: both numpy and PyTorch support arrays/tensors with zeros
        // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
        // `arr_sz` zero.
        unsigned int attr_val = 0;
        uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
            &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            (void *)arg_buffers[i]);

        if (ret_code != CUDA_SUCCESS || attr_val != CU_MEMORYTYPE_DEVICE) {
          // Copy to device buffer if arg is on host
          // - ret_code != CUDA_SUCCESS:
          //   arg_buffers[i] is not on device
          // - attr_val != CU_MEMORYTYPE_DEVICE:
          //   Cuda driver is aware of arg_buffers[i] but it might be on
          //   host.
          // See CUDA driver API `cuPointerGetAttribute` for more details.
          transferred = true;

          DeviceAllocation devalloc = executor->allocate_memory_ndarray(
              arr_sz, (uint64 *)device_result_buffer);
          device_buffers[i] = executor->get_ndarray_alloc_info_ptr(devalloc);
          temporary_devallocs[i] = devalloc;

          CUDADriver::get_instance().memcpy_host_to_device(
              (void *)device_buffers[i], arg_buffers[i], arr_sz);
        } else {
          device_buffers[i] = arg_buffers[i];
        }
        // device_buffers[i] saves a raw ptr on CUDA device.
        ctx.set_arg(i, (uint64)device_buffers[i]);

      } else if (arr_sz > 0) {
        // arg_buffers[i] is a DeviceAllocation*
        // TODO: Unwraps DeviceAllocation* can be done at TaskCodeGenLLVM
        // since it's shared by cpu and cuda.
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(arg_buffers[i]);
        device_buffers[i] = executor->get_ndarray_alloc_info_ptr(*ptr);
        // We compare arg_buffers[i] and device_buffers[i] later to check
        // if transfer happened.
        // TODO: this logic can be improved but I'll leave it to a followup
        // PR.
        arg_buffers[i] = device_buffers[i];

        // device_buffers[i] saves the unwrapped raw ptr from arg_buffers[i]
        ctx.set_arg(i, (uint64)device_buffers[i]);
      }
    }
  }
  if (transferred) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer,
                                            ctx.arg_buffer_size, nullptr);
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_arg_buffer, ctx.get_context().arg_buffer, ctx.arg_buffer_size,
        nullptr);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  for (auto task : offloaded_tasks) {
    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    cuda_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                        {&ctx.get_context()}, {});
  }
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().mem_free_async(device_arg_buffer, nullptr);
  }
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size,
        nullptr);
  }
  CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  // copy data back to host
  if (transferred) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (int i = 0; i < (int)parameters.size(); i++) {
      if (device_buffers[i] != arg_buffers[i]) {
        CUDADriver::get_instance().memcpy_device_to_host(
            arg_buffers[i], (void *)device_buffers[i],
            ctx.array_runtime_sizes[i]);
        executor->deallocate_memory_ndarray(temporary_devallocs[i]);
      }
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);

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

}  // namespace cuda
}  // namespace taichi::lang
