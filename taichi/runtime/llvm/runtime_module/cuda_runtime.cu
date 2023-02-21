#include <cuda_fp16.h>

extern "C" {

__device__ void half2_atomic_add(void *ptr_i, void *old_val_i, void *val_i) {
  half *ptr = (half *)ptr_i;
  half *old_val = (half *)old_val_i;
  half *val = (half *)val_i;

  __half2 v = {val[0], val[1]};
  __half2 old_v = atomicAdd((__half2 *)&ptr[0], v);

  *(__half2 *)&old_val[0] = old_v;
}
}
