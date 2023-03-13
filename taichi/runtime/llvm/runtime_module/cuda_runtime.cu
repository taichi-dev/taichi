#include <cuda_fp16.h>

/*
    Enable optimizations via hand-writting cuda kernels

    CUDA kernels defined in this file will be compiled by clang++ following:
   https://llvm.org/docs/CompileCudaWithLLVM.html. The resulting LLVM module
   will get loaded via "module_from_bitcode_file" and link with the LLVM Module
   from Taichi JIT compiler.

    During code generation, one may apply call("half2_atomic_add", ...) to make
   use of the kernels.
*/

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
