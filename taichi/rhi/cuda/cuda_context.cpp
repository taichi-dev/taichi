#define TI_RUNTIME_HOST
#include "cuda_context.h"

#include <unordered_map>
#include <mutex>

#include "taichi/system/threading.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_profiler.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

CUDAContext::CUDAContext()
    : profiler_(nullptr), driver_(CUDADriver::get_instance_without_context()) {
  // CUDA initialization
  dev_count_ = 0;
  driver_.init(0);
  driver_.device_get_count(&dev_count_);
  driver_.device_get(&device_, 0);

  char name[128];
  driver_.device_get_name(name, 128, device_);

  TI_TRACE("Using CUDA device [id=0]: {}", name);

  int cc_major, cc_minor;
  driver_.device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_);
  driver_.device_get_attribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_);

  int device_supports_mem_pool = 0;
  if (driver_.get_version_major() > 11 ||
      (driver_.get_version_major() == 11 && driver_.get_version_minor() >= 2)) {
    driver_.device_get_attribute(&device_supports_mem_pool,
                                 CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                                 device_);
  } else {
    TI_WARN(
        "Please consider upgrade your nvidia driver for better device memory "
        "pool"
        "support. Current driver supports CUDA {}.{}, we recommend driver "
        "version"
        "above 470 (CUDA 11.2).",
        driver_.get_version_major(), driver_.get_version_minor());
    device_supports_mem_pool = 0;
  }

  if (device_supports_mem_pool) {
    supports_mem_pool_ = true;
    void *default_mem_pool;
    driver_.device_get_default_mem_pool(&default_mem_pool, device_);
    // Let the memory pool have 128MB
    // TODO: make this configurable after we let the memory pool replace the
    // preallocated memory
    constexpr uint64 kMemPoolReleaseThreshold = 1048576 * 128;
    driver_.mem_pool_set_attribute(default_mem_pool,
                                   CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                   (void *)&kMemPoolReleaseThreshold);
  }

  TI_TRACE("CUDA Device Compute Capability: {}.{}", cc_major, cc_minor);
  driver_.primary_context_retain(&context_, 0);
  driver_.context_set_current(context_);

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  compute_capability_ = cc_major * 10 + cc_minor;

  if (compute_capability_ > 86) {
    compute_capability_ = 86;
  }

  driver_.device_get_attribute(
      &max_shared_memory_bytes_,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device_);

  mcpu_ = fmt::format("sm_{}", compute_capability_);

  TI_TRACE("Emitting CUDA code for {}", mcpu_);
}

std::size_t CUDAContext::get_total_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&_, &ret);
  return ret;
}

std::size_t CUDAContext::get_free_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&ret, &_);
  return ret;
}

std::string CUDAContext::get_device_name() {
  constexpr uint32_t kMaxNameStringLength = 128;
  char name[kMaxNameStringLength];
  driver_.device_get_name(name, kMaxNameStringLength /*=128*/, device_);
  std::string str(name);
  return str;
}

void CUDAContext::launch(void *func,
                         const std::string &task_name,
                         std::vector<void *> arg_pointers,
                         std::vector<int> arg_sizes,
                         unsigned grid_dim,
                         unsigned block_dim,
                         std::size_t dynamic_shared_mem_bytes) {
  // It is important to keep a handle since in async mode (deleted)
  // a constant folding kernel may happen during a kernel launch
  // then profiler->start and profiler->stop mismatch.
  // TODO: should we keep the handle?

  KernelProfilerBase::TaskHandle task_handle;
  // Kernel launch
  if (profiler_) {
    KernelProfilerCUDA *profiler_cuda =
        dynamic_cast<KernelProfilerCUDA *>(profiler_);
    std::string primal_task_name, key;
    bool valid =
        offline_cache::try_demangle_name(task_name, primal_task_name, key);
    profiler_cuda->trace(task_handle, valid ? primal_task_name : task_name,
                         func, grid_dim, block_dim, 0);
  }

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
    std::lock_guard<std::mutex> _(lock_);
    if (dynamic_shared_mem_bytes > 0) {
      if (dynamic_shared_mem_bytes > max_shared_memory_bytes_) {
        TI_ERROR(
            "Requested dynamic shared memory size of {} bytes, but the device "
            "supports max capacity of {} bytes.",
            dynamic_shared_mem_bytes, max_shared_memory_bytes_);
      }
      driver_.kernel_set_attribute(
          func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          dynamic_shared_mem_bytes);
    }
    driver_.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                          dynamic_shared_mem_bytes, nullptr,
                          arg_pointers.data(), nullptr);
  }
  if (profiler_)
    profiler_->stop(task_handle);

  if (debug_) {
    driver_.stream_synchronize(nullptr);
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

}  // namespace taichi::lang
