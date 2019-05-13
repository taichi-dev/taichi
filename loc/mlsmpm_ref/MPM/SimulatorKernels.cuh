#ifndef __SIMULATOR_KERNELS_CUH_
#define __SIMULATOR_KERNELS_CUH_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <Setting.h>
namespace mn {
__global__ void calcMaxVel(const int numParticle,
                           const T **d_vel,
                           T *_maxVelSquared);  ///< for time integration
__global__ void initMatrix(const int numParticle, T *d_F);
}  // namespace mn
#endif
