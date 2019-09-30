#ifndef __PARTICLE_KERNELS_CUH_
#define __PARTICLE_KERNELS_CUH_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <Setting.h>
namespace mn {
__global__ void aosToSoa(const int numParticle, T *_AoS, T **_SoA);
__global__ void soaToAos(const int numParticle, T **_SoA, T *_AoS);
__global__ void registerPage(const int numParticle,
                             const int tableSize,
                             const uint64_t *d_particleOffsets,
                             uint64_t *d_keyTable,
                             int *d_valTable,
                             int *d_numBucket);
__global__ void findPage(const int numParticle,
                         const int tableSize,
                         const uint64_t *d_particleOffsets,
                         uint64_t *d_keyTable,
                         int *d_valTable,
                         int *d_particle2bucket,
                         int *d_bucketSizes);
__global__ void reorderKey(const int numParticle,
                           const int *d_pageIds,
                           const int *d_offsets,
                           int *d_sizes,
                           const uint64_t *d_keys,
                           uint64_t *d_orderedKeys,
                           int *d_orderedIndices);
// need optimization
__global__ void calcOffset(const int numParticle,
                           const T one_over_dx,
                           const ulonglong3 _masks,
                           T **_pos,
                           uint64_t *_offsets);
__global__ void gather3D(int numParticle,
                         const int *_map,
                         const T **_ori,
                         T **_ord);
__global__ void gather3DShared(int numParticle,
                               const int *_map,
                               const T **_ori,
                               T **_ord);
__global__ void updateIndices(const int numParticle,
                              int *d_indices_old,
                              int *d_indices_map,
                              int *d_indices_new);
}  // namespace mn
#endif
