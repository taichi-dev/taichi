#pragma once
#if defined(TLANG_WITH_CUDA)

#include <taichi/tlang_util.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define check_cuda_errors(err) do {                           \
  CUresult __err = (err);                                     \
  if (int(__err))                                             \
    TC_ERROR("Cuda Error {}: {}", get_cuda_error_name(__err), \
             get_cuda_error_string(__err));                   \
  } while (0)

TLANG_NAMESPACE_BEGIN

inline std::string get_cuda_error_name(CUresult err) {
  const char *ptr;
  cuGetErrorName(err, &ptr);
  return std::string(ptr);
}

inline std::string get_cuda_error_string(CUresult err) {
  const char *ptr;
  cuGetErrorString(err, &ptr);
  return std::string(ptr);
}

inline std::string get_cuda_error_name(cudaError_t err) {
  const char *ptr = cudaGetErrorName(err);
  return std::string(ptr);
}

inline std::string get_cuda_error_string(cudaError_t err) {
  const char *ptr = cudaGetErrorString(err);
  return std::string(ptr);
}

#endif
TLANG_NAMESPACE_END
