#include "ParticleKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <cstdio>
namespace mn {
__global__ void soaToAos(const int numParticle, T **_SoA, T *_AoS) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  _AoS[3 * idx + 0] = _SoA[0][idx];
  _AoS[3 * idx + 1] = _SoA[1][idx];
  _AoS[3 * idx + 2] = _SoA[2][idx];
}
__global__ void aosToSoa(const int numParticle, T *_AoS, T **_SoA) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  _SoA[0][idx] = _AoS[3 * idx + 0];
  _SoA[1][idx] = _AoS[3 * idx + 1];
  _SoA[2][idx] = _AoS[3 * idx + 2];
}
__global__ void registerPage(const int numParticle,
                             const int tableSize,
                             const uint64_t *d_particleOffsets,
                             uint64_t *d_keyTable,
                             int *d_valTable,
                             int *d_numBucket) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  unsigned long long key = d_particleOffsets[idx] >> 12;
  unsigned long long hashkey = key % tableSize;
  unsigned long long ori;
  while ((ori = d_keyTable[hashkey]) != key) {
    if (ori == 0xffffffffffffffff)
      ori = atomicCAS((unsigned long long *)d_keyTable + hashkey,
                      0xffffffffffffffff,
                      key);  ///< -1 is the default value, means unoccupied
    if (d_keyTable[hashkey] ==
        key) {  ///< haven't found related record, so create a new entry
      if (ori == 0xffffffffffffffff)
        d_valTable[hashkey] = atomicAdd(d_numBucket, 1);  ///< created a record
      break;
    }
    hashkey += 127;  ///< search next entry
    if (hashkey >= tableSize)
      hashkey = hashkey % tableSize;
  }
}
__global__ void findPage(const int numParticle,
                         const int tableSize,
                         const uint64_t *d_particleOffsets,
                         uint64_t *d_keyTable,
                         int *d_valTable,
                         int *d_particle2bucket,
                         int *d_bucketSizes) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  unsigned long long key = d_particleOffsets[idx] >> 12;
  unsigned long long hashkey = key % tableSize;
  while (d_keyTable[hashkey] != key) {
    hashkey += 127;  ///< search next entry
    if (hashkey >= tableSize)
      hashkey = hashkey % tableSize;
  }
  int bucketid = d_valTable[hashkey];
  int cellid = (d_particleOffsets[idx] & 0xfc) >> 2;
  bucketid = (bucketid << 6) | cellid;
  d_particle2bucket[idx] = bucketid;
  atomicAdd(d_bucketSizes + bucketid, 1);
}
__global__ void reorderKey(const int numParticle,
                           const int *d_particle2bucket,
                           const int *d_offsets,
                           int *d_sizes,
                           const uint64_t *d_keys,
                           uint64_t *d_orderedKeys,
                           int *d_orderedIndices) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  int bucketid = d_particle2bucket[idx];
  d_orderedKeys[bucketid = (d_offsets[bucketid] +
                            atomicAdd(d_sizes + bucketid, 1))] = d_keys[idx];
  d_orderedIndices[bucketid] = idx;
}
__global__ void calcOffset(const int numParticle,
                           const T one_over_dx,
                           const ulonglong3 _masks,
                           T **_pos,
                           uint64_t *_offsets) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  _offsets[idx] |=
      Bit_Spread_Mine(_masks.x, (int)((_pos[0][idx]) * one_over_dx + 0.5f) - 1);
  _offsets[idx] |=
      Bit_Spread_Mine(_masks.y, (int)((_pos[1][idx]) * one_over_dx + 0.5f) - 1);
  _offsets[idx] |=
      Bit_Spread_Mine(_masks.z, (int)((_pos[2][idx]) * one_over_dx + 0.5f) - 1);
}
__global__ void gather3D(int numParticle,
                         const int *_map,
                         const T **_ori,
                         T **_ord) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  for (int i = 0; i < 3; i++)
    _ord[i][idx] = _ori[i][_map[idx]];
}
__global__ void gather3DShared(int numParticle,
                               const int *_map,
                               const T **_ori,
                               T **_ord) {
  extern __shared__ T s_Vals[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  for (int i = 0; i < 3; i++)
    s_Vals[(i << 9) + threadIdx.x] = _ori[i][_map[idx]];
  __syncthreads();
  for (int i = 0; i < 3; i++)
    _ord[i][idx] = s_Vals[(i << 9) + threadIdx.x];
}
__global__ void updateIndices(const int numParticle,
                              int *d_indices_old,
                              int *d_indices_map,
                              int *d_indices_new) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  d_indices_new[idx] = d_indices_old[d_indices_map[idx]];
}
}  // namespace mn
