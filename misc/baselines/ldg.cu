__global__ void test_ldg(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = __ldg(&b[i]) + __ldg((double *)&b[i]) + __ldg((char *)&b[i]);
}

int main() {
  float *a, *b;
  test_ldg<<<1, 1>>>(a, b);
}
