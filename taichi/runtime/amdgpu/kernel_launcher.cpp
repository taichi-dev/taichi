#include "taichi/runtime/amdgpu/kernel_launcher.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"

namespace taichi::lang {
namespace amdgpu {

bool KernelLauncher::on_amdgpu_device(void *ptr) {
  unsigned int attr_val[8];
  // mem_get_attribute doesn't work well on ROCm
  uint32_t ret_code =
      AMDGPUDriver::get_instance().mem_get_attributes.call(attr_val, ptr);

  return ret_code == HIP_SUCCESS && attr_val[0] == HIP_MEMORYTYPE_DEVICE;
}

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

  std::unordered_map<std::vector<int>, std::pair<void *, DeviceAllocation>,
                     hashing::Hasher<std::vector<int>>>
      transfers;
  std::unordered_map<std::vector<int>, void *,
                     hashing::Hasher<std::vector<int>>>
      device_ptrs;

  char *device_result_buffer{nullptr};
  AMDGPUDriver::get_instance().malloc(
      (void **)&device_result_buffer,
      std::max(ctx.result_buffer_size, sizeof(uint64)));

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &key = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[key];
      if (arr_sz == 0)
        continue;
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      std::vector<int> grad_ptr_idx = key;
      grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);

      if (ctx.device_allocation_type[key] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (on_amdgpu_device(data_ptr)) {
          device_ptrs[data_ptr_idx] = data_ptr;
        } else {
          DeviceAllocation devalloc = executor->allocate_memory_on_device(
              arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] =
              executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          AMDGPUDriver::get_instance().memcpy_host_to_device(
              (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
        }
        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
      } else if (arr_sz > 0) {  // why use arr_sz constrain?
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
      }
    } else if (parameter.is_argpack) {
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
      auto *argpack = ctx.argpack_ptrs[key];
      auto argpack_ptr = argpack->get_device_allocation();
      device_ptrs[data_ptr_idx] =
          executor->get_device_alloc_info_ptr(argpack_ptr);
      if (key.size() == 1) {
        ctx.set_argpack_ptr(key, (uint64)device_ptrs[data_ptr_idx]);
      } else {
        auto key_parent = key;
        key_parent.pop_back();
        auto *argpack_parent = ctx.argpack_ptrs[key_parent];
        argpack_parent->set_arg_nested_argpack_ptr(
            key.back(), (uint64)device_ptrs[data_ptr_idx]);
      }
    }
  }
  if (transfers.size() > 0) {
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    // Malloc_Async and Free_Async are available after ROCm 5.4
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
  if (transfers.size()) {
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      auto arg_id = idx;
      arg_id.pop_back();
      AMDGPUDriver::get_instance().memcpy_device_to_host(
          itr->second.first, (void *)device_ptrs[idx],
          ctx.array_runtime_sizes[arg_id]);
      executor->deallocate_memory_on_device(itr->second.second);
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
