extern "C" {

__device__ void block_barrier() {
  __syncthreads();
}
}
