#ifndef __DOMAIN_TRANSFORM_KERNELS_CUH_
#define __DOMAIN_TRANSFORM_KERNELS_CUH_
#include <cuda_runtime.h>
#include <stdint.h>
namespace mn {
__global__ void markPageBoundary(const int numParticle,
                                 const uint64_t *_offsets,
                                 int32_t *_marks);
__global__ void markCellBoundary(const int numParticle,
                                 const uint64_t *_offsets,
                                 int32_t *_marks);
__global__ void markBlockOffset(const int numParticle,
                                int32_t *_blockMap,
                                int32_t *_particleOffsets);
__global__ void markPageSize(const int numPage,
                             const int32_t *_particleOffsets,
                             int32_t *_marks);
__global__ void markVirtualPageOffset(const int numPage,
                                      const int32_t *_toVirtualOffsets,
                                      int32_t *_marks);
}  // namespace mn
#endif