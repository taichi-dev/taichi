#define TI_RUNTIME_HOST
#include "cuda_context.h"

#include <unordered_map>
#include <mutex>

#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/threading.h"
#include "taichi/backends/cuda/cuda_driver.h"

TLANG_NAMESPACE_BEGIN

CUDAContext::CUDAContext()
    : profiler(nullptr), driver(CUDADriver::get_instance_without_context()) {
  // CUDA initialization
  dev_count = 0;
  driver.init(0);
  driver.device_get_count(&dev_count);
  driver.device_get(&device, 0);

  char name[128];
  driver.device_get_name(name, 128, device);

  TI_TRACE("Using CUDA device [id=0]: {}", name);

  int cc_major, cc_minor;
  driver.device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  driver.device_get_attribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

  TI_TRACE("CUDA Device Compute Capability: {}.{}", cc_major, cc_minor);
  driver.context_create(&context, 0, device);

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  compute_capability = cc_major * 10 + cc_minor;

  if (compute_capability > 75) {
    // The NVPTX backend of LLVM 10.0.0 does not seem to support
    // compute_capability > 75 yet. See
    // llvm-10.0.0.src/build/lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc
    compute_capability = 75;
  }

  mcpu = fmt::format("sm_{}", compute_capability);

  TI_TRACE("Emitting CUDA code for {}", mcpu);
}

std::size_t CUDAContext::get_total_memory() {
  std::size_t ret, _;
  driver.mem_get_info(&_, &ret);
  return ret;
}

std::size_t CUDAContext::get_free_memory() {
  std::size_t ret, _;
  driver.mem_get_info(&ret, &_);
  return ret;
}

void CUDAContext::launch(void *func,
                         const std::string &task_name,
                         std::vector<void *> arg_pointers,
                         unsigned grid_dim,
                         unsigned block_dim,
                         std::size_t shared_mem_bytes) {
  // It is important to keep a handle since in async mode
  // a constant folding kernel may happen during a kernel launch
  // then profiler->start and profiler->stop mismatch.

  KernelProfilerBase::TaskHandle task_handle;
  // Kernel launch
  if (profiler)
    task_handle = profiler->start_with_handle(task_name);
  auto context_guard = CUDAContext::get_instance().get_guard();

  // TODO: remove usages of get_current_program here.
  // Make sure there are not too many threads for the device.
  // Note that the CUDA random number generator does not allow more than
  // [saturating_grid_dim * max_block_dim] threads.

  // These asserts are currently remove so that when GGUI calls CUDA kernels,
  // the grid and block dim won't be limited by the limits set by Program. With
  // these limits, GGUI would have to use kernels with grid strided loops, which
  // is harmful to performance. A simple example of rendering a bunny can drop
  // from 2000FPS to 1000FPS because of this. TI_ASSERT(grid_dim <=
  // get_current_program().config.saturating_grid_dim); TI_ASSERT(block_dim <=
  // get_current_program().config.max_block_dim);

  if (grid_dim > 0) {
    std::lock_guard<std::mutex> _(lock);
    driver.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                         shared_mem_bytes, nullptr, arg_pointers.data(),
                         nullptr);
  }
  if (profiler)
    profiler->stop(task_handle);

  if (debug) {
    driver.stream_synchronize(nullptr);
  }
}

CUDAContext::~CUDAContext() {
  // TODO: restore these?
  /*
  CUDADriver::get_instance().cuMemFree(context_buffer);
  for (auto cudaModule: cudaModules)
      CUDADriver::get_instance().cuModuleUnload(cudaModule);
  CUDADriver::get_instance().cuCtxDestroy(context);
  */
}

CUDAContext &CUDAContext::get_instance() {
  static auto context = new CUDAContext();
  return *context;
}

TLANG_NAMESPACE_END
