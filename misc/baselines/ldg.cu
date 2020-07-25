// Compile this file with clang to see how CUDA
// is translated into NVVM IR.

__device__ int cube(int x) {
  int y;
  asm(".reg .u32 t1;\n\t"            // temp reg t1
      " mul.lo.u32 t1, %1, %1;\n\t"  // t1 = x * x
      " mul.lo.u32 %0, t1, %1;"      // y = t1 * x
      : "=r"(y)
      : "r"(x));
  return y + clock64();
}

__global__ void __launch_bounds__(1024, 2) test_ldg(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = cube(1);
  auto c = __ldg((float4 *)a);
  *a += __ffs(__ballot_sync(__activemask(), 1));
  a[i] = __ldg(&b[i]) + __ldg((double *)&b[i]) + __ldg((char *)&b[i]) +
         __shfl_down_sync(__activemask(), i, 3);
}

int main() {
  float *a, *b;
  test_ldg<<<1, 1>>>(a, b);
}
