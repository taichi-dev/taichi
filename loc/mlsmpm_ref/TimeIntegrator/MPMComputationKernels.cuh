#ifndef __MPM_COMPUTATION_KERNELS_CUH_
#define __MPM_COMPUTATION_KERNELS_CUH_
#include <cuda_runtime.h>
#include <stdint.h>
#include <Setting.h>
namespace mn {
__global__ void calcIndex(const int numCell,
                          const T one_over_dx,
                          const int *d_cell_first_particles_indices,
                          const T **d_sorted_positions,
                          int **smallest_nodes);
__global__ void collideWithGround(ulonglong3 masks,
                                  uint64_t *pageOffsets,
                                  T **d_channels);
__global__ void SVD3_Pre_Calculation(int numparticle,
                                     float *input,
                                     float *ouputdata);
// compute stress
__global__ void computeContributionFixedCorotated(const int numParticle,
                                                  const T *d_F,
                                                  const T lambda,
                                                  const T mu,
                                                  const T volume,
                                                  T *d_contribution);
}  // namespace mn
#endif
