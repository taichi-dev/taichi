#ifndef __GRID_UPDATE_KERNELS_CUH_
#define __GRID_UPDATE_KERNELS_CUH_
#include <cuda_runtime.h>
#include <stdint.h>
#include <Setting.h>
namespace mn {
/// const int numTotalPage,
__global__ void applyGravity(const T dt, T **d_channels);
__global__ void updateVelocity(const T dt, T **d_channels);
__global__ void postP2G(T **d_channels);
__global__ void resetGrid(T **d_channels);
__global__ void preG2P(T **d_channels);
__global__ void setFlags(ulonglong3 masks,
                         uint64_t *pageOffsets,
                         unsigned *d_implicit_flags);
__global__ void implicitCopy(const int input_channel,
                             const T **input,
                             const int output_channel,
                             T **output);
__global__ void implicitScale(const T scale,
                              const T **input1,
                              const T **input2,
                              T **output);
__global__ void implicitMinus(const T **input, T **inoutput);
__global__ void implicitProject(const unsigned *flags, T **output);
__global__ void implicitClear(T **output);
__global__ void implicitSystemMatrix(const T scaled_dt_squared,
                                     T **d_channels,
                                     T **x,
                                     T **f);
__global__ void implicitConvergenceNorm(const T **d_implicit_x,
                                        T *_maxNormSquared);
__global__ void implicitInnerProduct(const T **d_channels,
                                     const T **d_implicit_x,
                                     const T **d_implicit_y,
                                     double *_innerProduct,
                                     int N);
}  // namespace mn
#endif
