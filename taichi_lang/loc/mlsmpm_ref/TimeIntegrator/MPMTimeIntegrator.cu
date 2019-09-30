#include "TimeIntegrator.cuh"
#include <MnBase/Math/Matrix/MatrixKernels.cuh>
#include "MPMComputationKernels.cuh"
#include "GridUpdateKernels.cuh"
#include "P2GKernels.cuh"
#include "G2PKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <System/CudaDevice/CudaKernelLauncher.cu>
namespace mn {
MPMTimeIntegrator::MPMTimeIntegrator(int transferScheme,
                                     int numParticle,
                                     T *dMemTrunk)
    : _transferScheme(transferScheme), d_memTrunk(dMemTrunk) {
  checkCudaErrors(
      cudaMalloc((void **)&d_contribution, sizeof(T) * numParticle * 9 + 2));
  d_tmp = d_memTrunk;
}
MPMTimeIntegrator::~MPMTimeIntegrator() {
  checkCudaErrors(cudaFree(d_contribution));
}
void MPMTimeIntegrator::transferP2G(
    const T dt,
    std::unique_ptr<Geometry> &geometry,
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  grid->clear();
#if TRANSFER_SCHEME == 0
  recordLaunch(
      std::string("P2G_FLIP"), (int)trans->_numVirtualPage, 512, (size_t)0,
      P2G_FLIP, geometry->_numParticle, (const int *)trans->d_targetPage,
      (const int *)trans->d_virtualPageOffset,
      (const int **)geometry->d_smallestNodeIndex, (const T *)d_contribution,
      trans->d_page2particle, trans->d_particle2cell, geometry->d_indices,
      geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_orderedMass,
      geometry->d_orderedVel, grid->d_channels, trans->d_adjPage);
#endif
#if TRANSFER_SCHEME == 1
  recordLaunch(
      std::string("P2G_APIC"), (int)trans->_numVirtualPage, 512, (size_t)0,
      P2G_APIC, geometry->_numParticle, (const int *)trans->d_targetPage,
      (const int *)trans->d_virtualPageOffset,
      (const int **)geometry->d_smallestNodeIndex, (const T *)d_contribution,
      trans->d_page2particle, trans->d_particle2cell, geometry->d_indices,
      geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_orderedMass,
      geometry->d_orderedVel, geometry->d_B, grid->d_channels,
      trans->d_adjPage);
#endif
#if TRANSFER_SCHEME == 2
  recordLaunch(
      std::string("P2G_MLS"), (int)trans->_numVirtualPage, 512, (size_t)0,
      P2G_MLS, geometry->_numParticle, (const int *)trans->d_targetPage,
      (const int *)trans->d_virtualPageOffset,
      (const int **)geometry->d_smallestNodeIndex, (const T *)d_contribution,
      trans->d_page2particle, trans->d_particle2cell, geometry->d_indices,
      geometry->d_indexTrans, geometry->d_orderedPos, geometry->d_orderedMass,
      geometry->d_orderedVel, geometry->d_B, dt, grid->d_channels,
      trans->d_adjPage, trans->d_pageOffset);
#endif
}
void MPMTimeIntegrator::call_postP2G(
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  recordLaunch(std::string("PostP2G"), trans->_numTotalPage, 64, (size_t)0,
               postP2G, grid->d_channels);
}
void MPMTimeIntegrator::call_resetGrid(
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  recordLaunch(std::string("ResetGrid"), trans->_numTotalPage, 64, (size_t)0,
               resetGrid, grid->d_channels);
}
void MPMTimeIntegrator::applyExternalForce(
    const T dt,
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  recordLaunch(std::string("ApplyGravity"), trans->_numTotalPage, 64, (size_t)0,
               applyGravity, dt, grid->d_channels);
}
void MPMTimeIntegrator::resolveCollision(
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  recordLaunch(
      std::string("CollideWithGround"), (int)trans->_numTotalPage, 64,
      (size_t)0, collideWithGround,
      make_ulonglong3(grid->h_masks[0], grid->h_masks[1], grid->h_masks[2]),
      trans->d_pageOffset, grid->d_channels);
}
void MPMTimeIntegrator::call_preG2P(
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  if (_transferScheme == 0)
    recordLaunch(std::string("PreG2P"), trans->_numTotalPage, 64, (size_t)0,
                 preG2P, grid->d_channels);
}
void MPMTimeIntegrator::transferG2P(
    const T dt,
    std::unique_ptr<Geometry> &geometry,
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
#if TRANSFER_SCHEME == 0
  recordLaunch(
      std::string("G2P_FLIP"), (int)trans->_numVirtualPage, 512, (size_t)0,
      G2P_FLIP, geometry->_numParticle, (const int *)trans->d_targetPage,
      (const int *)trans->d_virtualPageOffset,
      (const int **)geometry->d_smallestNodeIndex, trans->d_page2particle,
      trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans,
      geometry->d_orderedPos, geometry->d_orderedVel, grid->d_channels,
      geometry->d_F, d_tmp, dt, trans->d_adjPage);
  checkCudaErrors(cudaMemcpy(geometry->hd_orderedVel[0], d_tmp,
                             sizeof(T) * geometry->_numParticle,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(
      cudaMemcpy(geometry->hd_orderedVel[1], d_tmp + geometry->_numParticle,
                 sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
  checkCudaErrors(
      cudaMemcpy(geometry->hd_orderedVel[2], d_tmp + geometry->_numParticle * 2,
                 sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(geometry->d_F, d_tmp + geometry->_numParticle * 3,
                             sizeof(T) * geometry->_numParticle * 9,
                             cudaMemcpyDeviceToDevice));
#endif
#if TRANSFER_SCHEME == 1
  recordLaunch(
      std::string("G2P_APIC"), (int)trans->_numVirtualPage, 512, (size_t)0,
      G2P_APIC, geometry->_numParticle, (const int *)trans->d_targetPage,
      (const int *)trans->d_virtualPageOffset,
      (const int **)geometry->d_smallestNodeIndex, trans->d_page2particle,
      trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans,
      geometry->d_orderedPos, geometry->d_orderedVel, grid->d_channels,
      geometry->d_F, geometry->d_B, d_tmp, dt, trans->d_adjPage);
  checkCudaErrors(cudaMemcpy(geometry->hd_orderedVel[0], d_tmp,
                             sizeof(T) * geometry->_numParticle,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(
      cudaMemcpy(geometry->hd_orderedVel[1], d_tmp + geometry->_numParticle,
                 sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
  checkCudaErrors(
      cudaMemcpy(geometry->hd_orderedVel[2], d_tmp + geometry->_numParticle * 2,
                 sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(geometry->d_F, d_tmp + geometry->_numParticle * 3,
                             sizeof(T) * geometry->_numParticle * 9,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(geometry->d_B, d_tmp + geometry->_numParticle * 12,
                             sizeof(T) * geometry->_numParticle * 9,
                             cudaMemcpyDeviceToDevice));
#endif
#if TRANSFER_SCHEME == 2
  recordLaunch(
      std::string("G2P_MLS"), (int)trans->_numVirtualPage, 512, (size_t)0,
      G2P_MLS, geometry->_numParticle, (const int *)trans->d_targetPage,
      (const int *)trans->d_virtualPageOffset,
      (const int **)geometry->d_smallestNodeIndex, trans->d_page2particle,
      trans->d_particle2cell, geometry->d_indices, geometry->d_indexTrans,
      geometry->d_orderedPos, geometry->d_orderedVel, grid->d_channels,
      geometry->d_F, geometry->d_B, d_tmp, dt, trans->d_adjPage);
  checkCudaErrors(cudaMemcpy(geometry->hd_orderedVel[0], d_tmp,
                             sizeof(T) * geometry->_numParticle,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(
      cudaMemcpy(geometry->hd_orderedVel[1], d_tmp + geometry->_numParticle,
                 sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
  checkCudaErrors(
      cudaMemcpy(geometry->hd_orderedVel[2], d_tmp + geometry->_numParticle * 2,
                 sizeof(T) * geometry->_numParticle, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(geometry->d_B, d_tmp + geometry->_numParticle * 3,
                             sizeof(T) * geometry->_numParticle * 9,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(geometry->d_F, d_tmp + geometry->_numParticle * 12,
                             sizeof(T) * geometry->_numParticle * 9,
                             cudaMemcpyDeviceToDevice));
#endif
}
}  // namespace mn
