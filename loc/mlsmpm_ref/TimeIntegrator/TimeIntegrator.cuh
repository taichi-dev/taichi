#ifndef __TIME_INTEGRATOR_H_
#define __TIME_INTEGRATOR_H_
#include <Setting.h>
#include <Simulation/Model/Model.cuh>
#include <Simulation/Grid/GridDomain.cuh>
#include <Simulation/DomainTransform/DomainTransformer.cuh>
#include <memory>
namespace mn {
/// should think of a better runtime-polymorphism implementation rather than
/// inheritance
class MPMTimeIntegrator {
 public:
  MPMTimeIntegrator() = delete;
  MPMTimeIntegrator(int transferScheme, int numParticle, T *dMemTrunk);
  ~MPMTimeIntegrator();
  virtual void integrate(
      const T dt,
      Model &model,
      std::unique_ptr<SPGrid> &grid,
      std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans) = 0;
  void transferP2G(const T dt,
                   std::unique_ptr<Geometry> &geometry,
                   std::unique_ptr<SPGrid> &grid,
                   std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void call_postP2G(std::unique_ptr<SPGrid> &grid,
                    std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void call_resetGrid(
      std::unique_ptr<SPGrid> &grid,
      std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void applyExternalForce(
      const T dt,
      std::unique_ptr<SPGrid> &grid,
      std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void resolveCollision(
      std::unique_ptr<SPGrid> &grid,
      std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void call_preG2P(std::unique_ptr<SPGrid> &grid,
                   std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void transferG2P(const T dt,
                   std::unique_ptr<Geometry> &geometry,
                   std::unique_ptr<SPGrid> &grid,
                   std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);

 protected:
  const int _transferScheme;
  T *d_contribution;
  T *d_memTrunk;
  T *d_tmp;
};
///
class ExplicitTimeIntegrator : public MPMTimeIntegrator {
  friend class MPMSimulator;
 public:
  ExplicitTimeIntegrator(int transferScheme, int numParticle, T *dMemTrunk);
  ~ExplicitTimeIntegrator();
  virtual void integrate(
      const T dt,
      Model &model,
      std::unique_ptr<SPGrid> &grid,
      std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  void computeForceCoefficient(Model &model);
  void updateGridVelocity(
      const T dt,
      std::unique_ptr<SPGrid> &grid,
      std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> &trans);
  ///
 protected:
};
}  // namespace mn
#endif
