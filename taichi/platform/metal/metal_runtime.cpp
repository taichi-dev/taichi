#include "metal_runtime.h"

#include <algorithm>
#include <cstring>

#define TI_RUNTIME_HOST
#include <taichi/context.h>
#undef TI_RUNTIME_HOST

#ifdef TC_SUPPORTS_METAL

// If TC_SUPPORTS_METAL is defined, we are definitely on macOS. Therefore we
// don't need macro-guarding here.
#include <sys/mman.h>
#include <unistd.h>

TLANG_NAMESPACE_BEGIN

namespace metal {

namespace {
using KernelTaskType = OffloadedStmt::TaskType;

}  // namespace

BufferMemoryView::BufferMemoryView(size_t size, MemoryPool *mem_pool) {
  const size_t pagesize = getpagesize();
  // Both |ptr_| and |size_| must be aligned to page size.
  size_ = ((size + pagesize - 1) / pagesize) * pagesize;
  ptr_ = mem_pool->allocate(size_, pagesize);
  TC_ASSERT(ptr_ != nullptr);
}

MetalRuntime::CompiledMtlKernel::CompiledMtlKernel(
    const MetalKernelAttributes &md,
    MTLDevice *device,
    MTLFunction *func,
    CPUProfiler *profiler)
    : kernel_attribs_(md),
      pipeline_state_(new_compute_pipeline_state_with_function(device, func)),
      profiler_(profiler),
      profiler_id_(fmt::format("{}_dispatch", kernel_attribs_.name)) {
  TC_ASSERT(pipeline_state_ != nullptr);
}

MetalRuntime::CompiledTaichiKernel::CompiledTaichiKernel(
    const std::string &taichi_kernel_name,
    const std::string &source_code,
    const std::vector<MetalKernelAttributes> &mtl_kernels_attribs,
    size_t global_tmps_size,
    const MetalKernelArgsAttributes &args_attribs,
    MTLDevice *device,
    MemoryPool *mem_pool,
    CPUProfiler *profiler)
    : mtl_source_code_(source_code),
      global_tmps_mem_(global_tmps_size, mem_pool),
      args_attribs_(args_attribs),
      profiler_(profiler) {
  auto kernel_lib = new_library_with_source(device, mtl_source_code_);
  TC_ASSERT(kernel_lib != nullptr);
  for (const auto &ka : mtl_kernels_attribs) {
    auto kernel_func = new_function_with_name(kernel_lib.get(), ka.name);
    TC_ASSERT(kernel_func != nullptr);
    // Note that CompiledMtlKernel doesn't own |kernel_func|.
    compiled_mtl_kernels_.push_back(std::make_unique<CompiledMtlKernel>(
        ka, device, kernel_func.get(), profiler_));
  }
  global_tmps_buffer_ = new_mtl_buffer_no_copy(device, global_tmps_mem_.ptr(),
                                               global_tmps_mem_.size());
  if (args_attribs_.has_args()) {
    args_mem_ = std::make_unique<BufferMemoryView>(args_attribs_.total_bytes(),
                                                   mem_pool);
    args_buffer_ =
        new_mtl_buffer_no_copy(device, args_mem_->ptr(), args_mem_->size());
  }
}

MetalRuntime::MetalRuntime(size_t root_size,
                           MemoryPool *mem_pool,
                           CPUProfiler *profiler)
    : mem_pool_(mem_pool),
      profiler_(profiler),
      root_buffer_mem_(root_size, mem_pool) {
  device_ = mtl_create_system_default_device();
  TC_ASSERT(device_ != nullptr);
  command_queue_ = new_command_queue(device_.get());
  TC_ASSERT(command_queue_ != nullptr);
  create_new_command_buffer();
  root_buffer_ = new_mtl_buffer_no_copy(device_.get(), root_buffer_mem_.ptr(),
                                        root_buffer_mem_.size());
  TC_ASSERT(root_buffer_ != nullptr);
}

void MetalRuntime::register_taichi_kernel(
    const std::string &taichi_kernel_name,
    const std::string &mtl_kernel_source_code,
    const std::vector<MetalKernelAttributes> &kernels_attribs,
    size_t global_tmps_size,
    const MetalKernelArgsAttributes &args_attribs) {
  TC_ASSERT(compiled_taichi_kernels_.find(taichi_kernel_name) ==
            compiled_taichi_kernels_.end());
  TC_INFO("Registering taichi kernel \"{}\", Metal source code:\n{}",
          taichi_kernel_name, mtl_kernel_source_code);
  compiled_taichi_kernels_[taichi_kernel_name] =
      std::make_unique<CompiledTaichiKernel>(
          taichi_kernel_name, mtl_kernel_source_code, kernels_attribs,
          global_tmps_size, args_attribs, device_.get(), mem_pool_, profiler_);
}

void MetalRuntime::create_new_command_buffer() {
  cur_command_buffer_ = new_command_buffer(command_queue_.get());
  TC_ASSERT(cur_command_buffer_ != nullptr);
}

}  // namespace metal
TLANG_NAMESPACE_END

#endif  // TC_SUPPORTS_METAL
