#if defined(TLANG_WITH_CUDA)
#include "llvm_jit.h"
#include <cuda.h>
#include <taichi/context.h>

TLANG_NAMESPACE_BEGIN

class CUDAContext {
  CUdevice device;
  std::vector<CUmodule> cudaModules;
  CUcontext context;
  // CUlinkState linker;
  int devCount;
  CUdeviceptr context_buffer;
  std::string mcpu;

 public:
  CUDAContext();

  CUmodule compile(const std::string &ptx);

  CUfunction get_function(CUmodule module, const std::string &func_name);

  void launch(CUfunction func,
              void *context_ptr,
              unsigned gridDim,
              unsigned blockDim);

  std::string get_mcpu() const {
    return mcpu;
  }

  void make_current() {
    cuCtxSetCurrent(context);
  }

  ~CUDAContext();
};

extern std::unique_ptr<CUDAContext> cuda_context;

TLANG_NAMESPACE_END
#endif
