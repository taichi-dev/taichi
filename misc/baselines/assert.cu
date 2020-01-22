#include <cstdio>
#include <cassert>

__global__ void init_random_numbers(unsigned int seed) {
  printf("seed = %d\n", seed);
  atomicAdd(int *(12312433432), 123);
  assert(seed != 0);
}

int main() {
  init_random_numbers<<<1024, 1024>>>(1);
  return 0;
}
