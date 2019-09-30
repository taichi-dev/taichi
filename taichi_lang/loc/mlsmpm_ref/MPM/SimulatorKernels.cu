#include "SimulatorKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <stdio.h>
namespace mn {
__global__ void calcMaxVel(const int numParticle,
                           const T **d_vel,
                           T *_maxVelSquared) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  T vel_squared = d_vel[0][idx] * d_vel[0][idx];
  vel_squared += d_vel[1][idx] * d_vel[1][idx];
  vel_squared += d_vel[2][idx] * d_vel[2][idx];
  atomicMaxf(_maxVelSquared, vel_squared);
}
__global__ void initMatrix(const int numParticle, T *d_matrix) {
  int parid = blockDim.x * blockIdx.x + threadIdx.x;
  if (parid >= numParticle)
    return;
  for (int i = 0; i < 9; i++)
    d_matrix[parid + i * numParticle] = i % 3 == i / 3;
}
}  // namespace mn
