#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int m = 256;
constexpr int block_size = 128;

struct Node {
  int lock;
  int sum;
};

__global__ void inc(Node *nodes) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  int b = i * 3 % m;

  int warp_id = threadIdx.x % 32;
  int done = 0;
  auto mask = __activemask();
  // printf("mask %d\n", mask);
  while (!__all_sync(mask, done)) {
    for (int k = 0; k < 32; k++) {
      if (k == warp_id && !done) {
        int &lock = nodes[b].lock;
        if (atomicCAS(&lock, 0, 1) == 0) {
          nodes[b].sum += 1;
          done = true;
          atomicExch(&lock, 0);
        }
      }
    }
  }
}

int main() {
  Node *a;

  cudaMallocManaged(&a, m * sizeof(Node));

  for (int i = 0; i < 20; i++) {
    cudaDeviceSynchronize();
    auto t = get_time();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaDeviceSynchronize();
    inc<<<m, block_size>>>((Node *)a);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "device  " << milliseconds << std::endl;
    int sum = 0;
    for (int j = 0; j < m; j++) {
      sum += a[j].sum;
    }
    printf("sum %d\n", sum);
  }
  std::cout << std::endl;
}
