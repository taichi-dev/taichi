#include "GridKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <stdio.h>
namespace mn {
__global__ void buildHashMapFromPage(const int numPage,
                                     const int tableSize,
                                     const uint64_t *d_masks,
                                     const uint64_t *d_particleOffsets,
                                     const int *d_page2particle,
                                     uint64_t *d_keyTable,
                                     int *d_valTable,
                                     uint64_t *d_pageOffsets) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numPage)
    return;
  uint64_t key = d_particleOffsets[d_page2particle[idx]] >> 12;
  uint64_t hashkey = key % tableSize;
  d_pageOffsets[idx] = key << 12;
  do {
    atomicCAS((unsigned long long int *)d_keyTable + hashkey,
              0xffffffffffffffff,
              (unsigned long long int)
                  key);  ///< -1 is the default value, means unoccupied
    if (d_keyTable[hashkey] ==
        key) {  ///< haven't found related record, so create a new entry
      d_valTable[hashkey] = idx;  ///< created a record
      break;
    } else {
      hashkey += 127;  ///< search next entry
      if (hashkey >= tableSize)
        hashkey = hashkey % tableSize;
    }
  } while (true);
}
__global__ void supplementAdjacentPages(const int numPage,
                                        const int tableSize,
                                        const uint64_t *d_masks,
                                        const uint64_t *d_particleOffsets,
                                        const uint64_t *d_neighborOffsets,
                                        const int *d_page2particle,
                                        uint64_t *d_keyTable,
                                        int *d_valTable,
                                        int *d_totalPage,
                                        uint64_t *d_pageOffsets) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numPage)
    return;
  int sparsePageId;
  uint64_t okey = d_particleOffsets[d_page2particle[idx]] & 0xfffffffffffff000;
  for (int i = 0; i < 7; i++) {
    uint64_t key = Packed_Add(d_masks, okey, d_neighborOffsets[i]) >>
                   12;  ///< dense page id, used as key
    uint64_t hashkey = key % tableSize;
    while (d_keyTable[hashkey] != key) {
      uint64_t old = atomicCAS(
          (unsigned long long int *)d_keyTable + hashkey, 0xffffffffffffffff,
          (unsigned long long int)
              key);  ///< -1 is the default value, means unoccupied
      if (d_keyTable[hashkey] == key) {
        if (old == 0xffffffffffffffff) {  ///< created a new entry
          d_valTable[hashkey] = sparsePageId =
              atomicAdd(d_totalPage, 1);  ///< created a record
          d_pageOffsets[sparsePageId] = key << 12;
        }
        break;
      } else {
        hashkey += 127;  ///< search next entry
        if (hashkey >= tableSize)
          hashkey = hashkey % tableSize;
      }
    }
  }
}
__global__ void establishPageTopology(const int numPage,
                                      const int tableSize,
                                      const uint64_t *d_masks,
                                      const uint64_t *d_particleOffsets,
                                      const uint64_t *d_neighborOffsets,
                                      const int *d_page2particle,
                                      uint64_t *d_keyTable,
                                      int *d_valTable,
                                      int **d_adjPage) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numPage)
    return;
  uint64_t okey = d_particleOffsets[d_page2particle[idx]] & 0xfffffffffffff000;
  for (int i = 0; i < 7; i++) {
    uint64_t key = Packed_Add(d_masks, okey, d_neighborOffsets[i]) >>
                   12;  ///< dense page id, used as key
    uint64_t hashkey = key % tableSize;
    while (d_keyTable[hashkey] != key) {
      hashkey += 127;  ///< search next entry
      if (hashkey >= tableSize)
        hashkey = hashkey % tableSize;
    }
    d_adjPage[i][idx] = d_valTable[hashkey];
  }
}
}  // namespace mn
