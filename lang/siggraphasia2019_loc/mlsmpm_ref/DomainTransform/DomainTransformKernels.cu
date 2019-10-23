#include "DomainTransformKernels.cuh"
namespace mn {
// mark the boundary where the page # changes
__global__ void markPageBoundary(const int numParticle,
                                 const uint64_t *_offsets,
                                 int32_t *_marks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  if (!idx || (_offsets[idx] >> 12) != (_offsets[idx - 1] >> 12))
    _marks[idx] = 1;
  else
    _marks[idx] = 0;
  if (idx == 0)
    _marks[numParticle] = 1;
}
// mark the boundary where the cell # changes
__global__ void markCellBoundary(const int numParticle,
                                 const uint64_t *_offsets,
                                 int32_t *_marks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  if (!idx || _offsets[idx] != _offsets[idx - 1])
    _marks[idx] = 1;
  else
    _marks[idx] = 0;
  if (idx == 0)
    _marks[numParticle] = 1;
}
__global__ void markBlockOffset(const int numParticle,
                                int32_t *_blockMap,
                                int32_t *_particleOffsets) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numParticle)
    return;
  int mapid = _blockMap[idx];
  if (!idx || mapid != _blockMap[idx - 1]) {
    _particleOffsets[mapid - 1] = idx;
  }
  /// mark sentinel
  if (idx == numParticle - 1)
    _particleOffsets[mapid] = numParticle;
}
__global__ void markPageSize(const int numPage,
                             const int32_t *_offsets,
                             int32_t *_marks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPage)
    return;
  _marks[idx] = (_offsets[idx + 1] - _offsets[idx] + 511) / 512;
}
__global__ void markVirtualPageOffset(const int numPage,
                                      const int32_t *_toVirtualOffsets,
                                      int32_t *_marks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPage)
    return;
  _marks[_toVirtualOffsets[idx]] = 1;
}
}  // namespace mn
