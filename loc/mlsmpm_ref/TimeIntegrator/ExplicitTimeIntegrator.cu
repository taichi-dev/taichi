#include "TimeIntegrator.cuh"
#include <MnBase/Math/Matrix/MatrixKernels.cuh>
#include "MPMComputationKernels.cuh"
#include "GridUpdateKernels.cuh"
#include "P2GKernels.cuh"
#include "G2PKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <System/CudaDevice/CudaKernelLauncher.cu>
namespace mn {
ExplicitTimeIntegrator::ExplicitTimeIntegrator(int transferScheme,
                                               int numParticle,
                                               T *dMemTrunk)
    : MPMTimeIntegrator(transferScheme, numParticle, dMemTrunk) {
}
ExplicitTimeIntegrator::~ExplicitTimeIntegrator() {
}
void ExplicitTimeIntegrator::integrate(
    const T dt,
    Model &model,
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  auto &geometry = model.refGeometryPtr();
  trans->rebuild();
  Logger::recordSection<TimerType::GPU>("build_particle_grid_mapping");
  configuredLaunch({"CalcIndex", trans->_numCell}, calcIndex, trans->_numCell,
                   one_over_dx, (const int *)(trans->d_cell2particle),
                   (const T **)geometry->d_orderedPos,
                   geometry->d_smallestNodeIndex);
  Logger::recordSection<TimerType::GPU>("calc_smallest_index");
  computeForceCoefficient(model);
  Logger::recordSection<TimerType::GPU>(
      "contribution_calculation(include_svd)");
  transferP2G(dt, geometry, grid, trans);
  Logger::recordSection<TimerType::GPU>("p2g_calculation");
  call_postP2G(grid, trans);
  applyExternalForce(dt, grid, trans);
  updateGridVelocity(dt, grid, trans);
  resolveCollision(grid, trans);
  call_preG2P(grid, trans);
  Logger::recordSection<TimerType::GPU>("explicit_grid_update");
  transferG2P(dt, geometry, grid, trans);
  Logger::recordSection<TimerType::GPU>("g2p_calculation");
  call_resetGrid(grid, trans);
  geometry->reorder();
  Logger::recordSection<TimerType::GPU>("particle_reorder");
  Logger::blankLine<TimerType::GPU>();
}
void ExplicitTimeIntegrator::computeForceCoefficient(Model &model) {
  auto &geometry = model.refGeometryPtr();
  auto &refMaterialPtr = model.refMaterialDynamicsPtr();
  auto material = (ElasticMaterialDynamics *)refMaterialPtr.get();
  configuredLaunch(
      {"ComputeContributionFixedCorotated", geometry->_numParticle},
      computeContributionFixedCorotated, geometry->_numParticle,
      (const T *)geometry->d_F, material->_lambda, material->_mu,
      material->_volume, d_contribution);
}
/// explicit grid update Ft = mv
void ExplicitTimeIntegrator::updateGridVelocity(
    const T dt,
    std::unique_ptr<SPGrid> &grid,
    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) {
  recordLaunch(std::string("UpdateVelocity"), trans->_numTotalPage, 64,
               (size_t)0, updateVelocity, dt, grid->d_channels);
}
}  // namespace mn
