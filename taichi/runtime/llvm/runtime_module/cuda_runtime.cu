#include <cuda_fp16.h>

extern "C" {

__device__ void half2_atomic_add(half *ptr, half *old_val, const half val[2]) {
  __half2 v = {val[0], val[1]};
  __half2 old_v = atomicAdd((__half2 *)&ptr[0], v);

  *(__half2 *)&old_val[0] = old_v;
}
}
