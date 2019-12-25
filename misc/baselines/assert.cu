#include <cstdio>
#include <cassert>

// https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
__global__ void init_random_numbers(unsigned int seed) {
  printf("seed = %d\n", seed);
  assert(seed != 0);
}

int main() {
  init_random_numbers<<<1024, 1024>>>(1);
  return 0;
}
