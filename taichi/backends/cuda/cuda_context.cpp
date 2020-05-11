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

  mcpu = fmt::format("sm_{}{}", cc_major, cc_minor);
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
                         unsigned gridDim,
                         unsigned blockDim) {
  // Kernel launch
  if (profiler)
    profiler->start(task_name);
  auto context_guard = CUDAContext::get_instance().get_guard();
  if (gridDim > 0) {
    std::lock_guard<std::mutex> _(lock);
    driver.launch_kernel(func, gridDim, 1, 1, blockDim, 1, 1, 0, nullptr,
                         arg_pointers.data(), nullptr);
  }
  if (profiler)
    profiler->stop();

  if (get_current_program().config.debug) {
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
  static std::unordered_map<std::thread::id, CUDAContext *> instances;
  static std::mutex mut;
  {
    // critical section
    auto _ = std::lock_guard<std::mutex>(mut);

    auto tid = std::this_thread::get_id();
    if (instances.find(tid) == instances.end()) {
      instances[tid] = new CUDAContext();
      // We expect CUDAContext to live until the process ends, thus the raw
      // pointers and `new`s.
    }
    return *instances[tid];
  }
}

TLANG_NAMESPACE_END
