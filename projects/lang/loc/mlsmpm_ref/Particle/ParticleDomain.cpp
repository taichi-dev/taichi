#include "ParticleDomain.cuh"
#include <MnBase/Meta/AllocMeta.h>
#include <MnBase/AggregatedAttribs.h>
namespace mn {
Particles::Particles(int numParticle, int tableSize)
    : _numParticle(numParticle), _tableSize(tableSize) {
  reportMemory("before particle_allocation");
  /// allocate
  _attribs = cuda_allocs<ParticleAttribs>(_numParticle + 2);
  checkCudaErrors(cudaMalloc((void **)&d_indices, sizeof(int) * _numParticle));
  checkCudaErrors(
      cudaMalloc((void **)&d_indexTrans, sizeof(int) * _numParticle));
  checkCudaErrors(cudaMalloc((void **)&d_numBucket, sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&d_particle2bucket, sizeof(int) * _numParticle));
  checkCudaErrors(
      cudaMalloc((void **)&d_bucketSizes, sizeof(int) * _tableSize));
  checkCudaErrors(
      cudaMalloc((void **)&d_bucketOffsets, sizeof(int) * _tableSize));
  printf("num particle:%d, table size:%d\n", _numParticle, _tableSize);
  /// alias
  d_offsets = (uint64_t *)_attribs[(int)ParticleAttribIndex::OFFSET];
  d_mass = (T *)_attribs[(int)ParticleAttribIndex::MASS];
  d_orderedMass = (T *)_attribs[(int)ParticleAttribIndex::ORD_MASS];
  hd_pos[0] = (T *)_attribs[(int)ParticleAttribIndex::POSX];
  hd_pos[1] = (T *)_attribs[(int)ParticleAttribIndex::POSY];
  hd_pos[2] = (T *)_attribs[(int)ParticleAttribIndex::POSZ];
  hd_orderedPos[0] = (T *)_attribs[(int)ParticleAttribIndex::ORD_POSX];
  hd_orderedPos[1] = (T *)_attribs[(int)ParticleAttribIndex::ORD_POSY];
  hd_orderedPos[2] = (T *)_attribs[(int)ParticleAttribIndex::ORD_POSZ];
  hd_orderedVel[0] = (T *)_attribs[(int)ParticleAttribIndex::ORD_VELX];
  hd_orderedVel[1] = (T *)_attribs[(int)ParticleAttribIndex::ORD_VELY];
  hd_orderedVel[2] = (T *)_attribs[(int)ParticleAttribIndex::ORD_VELZ];
  hd_smallestNodeIndex[0] = (int *)_attribs[(int)ParticleAttribIndex::INDEXX];
  hd_smallestNodeIndex[1] = (int *)_attribs[(int)ParticleAttribIndex::INDEXY];
  hd_smallestNodeIndex[2] = (int *)_attribs[(int)ParticleAttribIndex::INDEXZ];
  checkCudaErrors(cudaMalloc((void **)&d_pos, sizeof(T *) * Dim));
  checkCudaErrors(cudaMalloc((void **)&d_orderedPos, sizeof(T *) * Dim));
  checkCudaErrors(cudaMalloc((void **)&d_orderedVel, sizeof(T *) * Dim));
  checkCudaErrors(
      cudaMalloc((void **)&d_smallestNodeIndex, sizeof(int *) * Dim));
  checkCudaErrors(
      cudaMemcpy(d_pos, hd_pos, sizeof(T *) * Dim, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_orderedPos, hd_orderedPos, sizeof(T *) * Dim,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_orderedVel, hd_orderedVel, sizeof(T *) * Dim,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_smallestNodeIndex, hd_smallestNodeIndex,
                             sizeof(int *) * Dim, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_min, sizeof(float3)));
  checkCudaErrors(cudaMalloc((void **)&d_F, sizeof(T) * 9 * _numParticle));
  checkCudaErrors(cudaMalloc((void **)&d_B, sizeof(T) * 9 * _numParticle));
  reportMemory("after particle_allocation");
}
Particles::~Particles() {
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_smallestNodeIndex));
  checkCudaErrors(cudaFree(d_orderedVel));
  checkCudaErrors(cudaFree(d_orderedPos));
  checkCudaErrors(cudaFree(d_pos));
  checkCudaErrors(cudaFree(d_indices));
  checkCudaErrors(cudaFree(d_indexTrans));
  checkCudaErrors(cudaFree(d_F));
  checkCudaErrors(cudaFree(d_B));
}
}  // namespace mn
