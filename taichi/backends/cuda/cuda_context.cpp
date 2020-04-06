#if defined(TI_WITH_CUDA)
#define TI_RUNTIME_HOST

#include <unordered_map>

#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/threading.h"

#include "cuda_context.h"

TLANG_NAMESPACE_BEGIN

CUDAContext::CUDAContext() : profiler(nullptr) {
  // CUDA initialization
  dev_count = 0;
  check_cuda_error(cuInit(0));
  check_cuda_error(cuDeviceGetCount(&dev_count));
  check_cuda_error(cuDeviceGet(&device, 0));

  char name[128];
  check_cuda_error(cuDeviceGetName(name, 128, device));

  TI_TRACE("Using CUDA device [id=0]: {}", name);

  int cc_major, cc_minor;
  check_cuda_error(cuDeviceGetAttribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  check_cuda_error(cuDeviceGetAttribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  TI_TRACE("CUDA Device Compute Capability: {}.{}", cc_major, cc_minor);
  check_cuda_error(cuCtxCreate(&context, 0, device));

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  mcpu = fmt::format("sm_{}{}", cc_major, cc_minor);
}

std::size_t CUDAContext::get_total_memory() {
  std::size_t ret, _;
  check_cuda_error(cuMemGetInfo(&_, &ret));
  return ret;
}

std::size_t CUDAContext::get_free_memory() {
  std::size_t ret, _;
  check_cuda_error(cuMemGetInfo(&ret, &_));
  return ret;
}

void CUDAContext::launch(void *func,
                         const std::string &task_name,
                         std::vector<void *> arg_pointers,
                         unsigned gridDim,
                         unsigned blockDim) {
  // auto _ = CUDAContext::get_instance().get_guard();
  make_current();

  // Kernel launch
  if (profiler)
    profiler->start(task_name);
  if (gridDim > 0) {
    std::lock_guard<std::mutex> _(lock);
    check_cuda_error(cuLaunchKernel((CUfunction)func, gridDim, 1, 1, blockDim,
                                    1, 1, 0, nullptr, arg_pointers.data(),
                                    nullptr));
  }
  if (profiler)
    profiler->stop();

  if (get_current_program().config.debug) {
    check_cuda_error(cuStreamSynchronize((CUstream)0));
  }
}

CUDAContext::~CUDAContext() {
  /*
  check_cuda_error(cuMemFree(context_buffer));
  for (auto cudaModule: cudaModules)
    check_cuda_error(cuModuleUnload(cudaModule));
  check_cuda_error(cuCtxDestroy(context));
  */
}

CUDAContext &CUDAContext::get_instance() {
  auto tid = std::this_thread::get_id();
  if (instances.find(tid) == instances.end()) {
    instances[tid] = std::make_unique<CUDAContext>();
  }
  return *instances[tid];
}

std::unordered_map<std::thread::id, std::unique_ptr<CUDAContext>>
    CUDAContext::instances;

TLANG_NAMESPACE_END
#endif
