// https://llvm.org/docs/CompileCudaWithLLVM.html
// clang++-7 experiment.cu --cuda-gpu-arch=sm_35 -std=c++14 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread

extern "C" __global__ void add(int *a, int *b, int *c) {
  // auto i = threadIdx.x;
  // c[i] = a[i] + b[i];
  printf("%d %f\n", 123, 456.0);
}

int main() {}