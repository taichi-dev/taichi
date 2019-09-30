#ifndef __GRID_KERNELS_CUH_
#define __GRID_KERNELS_CUH_
#include <cuda_runtime.h>
#include <stdint.h>
namespace mn {
/// grid
__global__ void buildHashMapFromPage(const int numPage,
                                     const int tableSize,
                                     const uint64_t *d_masks,
                                     const uint64_t *d_particleOffsets,
                                     const int *d_page2particle,
                                     uint64_t *d_keyTable,
                                     int *d_valTable,
                                     uint64_t *d_pageOffsets);
__global__ void supplementAdjacentPages(const int numPage,
                                        const int tableSize,
                                        const uint64_t *d_masks,
                                        const uint64_t *d_particleOffsets,
                                        const uint64_t *d_neighborOffsets,
                                        const int *d_page2particle,
                                        uint64_t *d_keyTable,
                                        int *d_valTable,
                                        int *d_totalPage,
                                        uint64_t *d_pageOffsets);
__global__ void establishPageTopology(const int numPage,
                                      const int tableSize,
                                      const uint64_t *d_masks,
                                      const uint64_t *d_particleOffsets,
                                      const uint64_t *d_neighborOffsets,
                                      const int *d_page2particle,
                                      uint64_t *d_keyTable,
                                      int *d_valTable,
                                      int **d_adjPage);
}  // namespace mn
#endif