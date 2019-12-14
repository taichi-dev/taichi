#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdio>

constexpr int num_states = 1024 * 1024;
__device__ curandState_t states[num_states];

// https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
__global__ void init_random_numbers(unsigned int seed) {
  int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                 (threadIdx.z * (blockDim.x * blockDim.y)) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  int idx = threadId;
  curand_init(idx + (seed * 1000000007), 0, 0, &states[idx]);
}

__device__ float randf() {
  int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                 (threadIdx.z * (blockDim.x * blockDim.y)) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  int idx = threadId % num_states;
  return curand_uniform(&states[idx]);
}

__global__ void print_rand() {
  printf("%f\n", randf());
}

int main() {
  init_random_numbers<<<1024, 1024>>>(1);
  print_rand<<<32, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}
