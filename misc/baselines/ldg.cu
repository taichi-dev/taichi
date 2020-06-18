__device__ int cube(int x) {
  int y;
  asm(".reg .u32 t1;\n\t"            // temp reg t1
      " mul.lo.u32 t1, %1, %1;\n\t"  // t1 = x * x
      " mul.lo.u32 %0, t1, %1;"      // y = t1 * x
      : "=r"(y)
      : "r"(x));
  return y;
}

__global__ void test_ldg(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = cube(1);
  auto c = __ldg((float4 *)a);
  a[i] = __ldg(&b[i]) + __ldg((double *)&b[i]) + __ldg((char *)&b[i]);
}

int main() {
  float *a, *b;
  test_ldg<<<1, 1>>>(a, b);
}
