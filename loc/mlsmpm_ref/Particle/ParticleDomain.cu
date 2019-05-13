#include "ParticleDomain.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <System/CudaDevice/CudaKernelLauncher.cu>
namespace mn {
void Particles::initialize(const T *h_mass,
                           const T *h_pos,
                           const T *h_vel,
                           const uint64_t *_dmasks,
                           const uint64_t *_hmasks,
                           uint64_t *keyTable,
                           int *valTable,
                           void *_memTrunk) {
  d_masks = _dmasks;
  h_masks = _hmasks;
  d_tmp = static_cast<T *>(_memTrunk);
  d_keyTable = keyTable;
  d_valTable = valTable;
  // copy in mass pos vel
  checkCudaErrors(cudaMemcpy(d_mass, h_mass, sizeof(T) * _numParticle,
                             cudaMemcpyHostToDevice));  // or use thrust::copy
  checkCudaErrors(cudaMemcpy(d_tmp, h_pos, sizeof(T) * _numParticle * Dim,
                             cudaMemcpyHostToDevice));  // or use thrust::copy
  configuredLaunch({"AosToSoa", _numParticle}, aosToSoa, _numParticle, d_tmp,
                   d_pos);
  // calculate offsets
  checkCudaErrors(cudaMemset(d_offsets, 0, sizeof(uint64_t) * _numParticle));
  configuredLaunch(
      {"CalcOffset", _numParticle}, calcOffset, _numParticle, one_over_dx,
      (const ulonglong3)make_ulonglong3(h_masks[0], h_masks[1], h_masks[2]),
      d_pos, d_offsets);
  thrust::sequence(getDevicePtr(d_indices),
                   getDevicePtr(d_indices) + _numParticle);
  sort_by_offsets();
  recordLaunch(std::string("Gather3DShared"), (_numParticle + 511) / 512, 512,
               sizeof(T) * 512 * 3, gather3DShared, _numParticle,
               (const int *)d_indices, (const T **)d_pos, d_orderedPos);
  checkCudaErrors(cudaMemcpy(d_tmp, h_vel, sizeof(T) * _numParticle * Dim,
                             cudaMemcpyHostToDevice));  // or use thrust::copy
  configuredLaunch({"AosToSoa", _numParticle}, aosToSoa, _numParticle, d_tmp,
                   d_orderedVel);
  checkCudaErrors(cudaMemcpy(d_orderedMass, d_mass, sizeof(T) * _numParticle,
                             cudaMemcpyDeviceToDevice));
}
void Particles::sort_by_offsets() {
  checkCudaErrors(cudaMemcpy((uint64_t *)d_tmp + _numParticle, d_indices,
                             sizeof(int) * _numParticle,
                             cudaMemcpyDeviceToDevice));
  /// histogram sort (substitution of radix sort)
  checkCudaErrors(cudaMemset(d_numBucket, 0, sizeof(int)));
  checkCudaErrors(cudaMemset(d_keyTable, 0xff, sizeof(uint64_t) * _tableSize));
  // to write for sorting optimization
  configuredLaunch({"RegisterPage", _numParticle}, registerPage, _numParticle,
                   _tableSize, (const uint64_t *)d_offsets, d_keyTable,
                   d_valTable, d_numBucket);
  checkCudaErrors(cudaMemcpy((void **)&_numBucket, d_numBucket, sizeof(int),
                             cudaMemcpyDeviceToHost));
  _numBucket <<= 6;
  printf("during particle ordering    num particle: %d, num bucket: %d(%d)\n",
         _numParticle, _numBucket, _numBucket >> 6);
  checkCudaErrors(cudaMemset(d_bucketSizes, 0, sizeof(int) * _numBucket));
  configuredLaunch({"FindPage", _numParticle}, findPage, _numParticle,
                   _tableSize, (const uint64_t *)d_offsets, d_keyTable,
                   d_valTable, d_particle2bucket, d_bucketSizes);
  Logger::tick<TimerType::GPU>();
  checkThrustErrors(thrust::exclusive_scan(
      getDevicePtr(d_bucketSizes), getDevicePtr(d_bucketSizes) + _numBucket,
      getDevicePtr(d_bucketOffsets)));
  Logger::tock<TimerType::GPU>("calc bucket offsets");
  checkCudaErrors(cudaMemset(d_bucketSizes, 0, sizeof(int) * _numBucket));
  checkCudaErrors(cudaMemcpy(d_tmp, d_offsets, sizeof(uint64_t) * _numParticle,
                             cudaMemcpyDeviceToDevice));
  configuredLaunch({"ReorderKey", _numParticle}, reorderKey, _numParticle,
                   (const int *)d_particle2bucket, (const int *)d_bucketOffsets,
                   d_bucketSizes, (const uint64_t *)d_tmp, d_offsets,
                   d_indices);
  // d_indexTrans from last timestep to current
  checkCudaErrors(cudaMemcpy(d_indexTrans, d_indices,
                             sizeof(int) * _numParticle,
                             cudaMemcpyDeviceToDevice));
  configuredLaunch({"UpdateIndices", _numParticle}, updateIndices, _numParticle,
                   (int *)d_tmp + 2 * _numParticle, d_indexTrans, d_indices);
}
void Particles::reorder() {
  checkCudaErrors(cudaMemset(d_offsets, 0, sizeof(uint64_t) * _numParticle));
  configuredLaunch(
      {"CalcOffset", _numParticle}, calcOffset, _numParticle, one_over_dx,
      (const ulonglong3)make_ulonglong3(h_masks[0], h_masks[1], h_masks[2]),
      d_orderedPos, d_offsets);
  sort_by_offsets();
  // position
  for (int i = 0; i < Dim; i++)
    checkCudaErrors(cudaMemcpy(hd_pos[i], hd_orderedPos[i],
                               sizeof(T) * _numParticle,
                               cudaMemcpyDeviceToDevice));
  recordLaunch(std::string("Gather3DShared"), (_numParticle + 511) / 512, 512,
               sizeof(T) * 512 * 3, gather3DShared, _numParticle,
               (const int *)d_indexTrans, (const T **)d_pos, d_orderedPos);
}
void Particles::retrieveAttribs(std::vector<std::array<T, Dim>> &h_pos,
                                std::vector<std::array<T, Dim>> &h_vel,
                                std::vector<int> &h_indices) {
  // velocity
  recordLaunch(std::string("Gather3DShared"), (_numParticle + 511) / 512, 512,
               sizeof(T) * 512 * 3, gather3DShared, _numParticle,
               (const int *)d_indexTrans, (const T **)d_orderedVel, d_pos);
  configuredLaunch({"SoaToAos", _numParticle}, soaToAos, _numParticle, d_pos,
                   d_tmp);
  checkCudaErrors(cudaMemcpy(&h_vel[0][0], d_tmp, sizeof(T) * _numParticle * 3,
                             cudaMemcpyDeviceToHost));
  // position
  configuredLaunch({"SoaToAos", _numParticle}, soaToAos, _numParticle,
                   d_orderedPos, d_tmp);
  checkCudaErrors(cudaMemcpy(&h_pos[0][0], d_tmp, sizeof(T) * _numParticle * 3,
                             cudaMemcpyDeviceToHost));
  // global indices
  checkCudaErrors(cudaMemcpy(&h_indices[0], d_indices, sizeof(T) * _numParticle,
                             cudaMemcpyDeviceToHost));
}
}  // namespace mn
