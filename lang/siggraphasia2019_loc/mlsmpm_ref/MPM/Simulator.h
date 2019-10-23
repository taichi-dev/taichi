#ifndef __MPM_SIMULATOR_H_
#define __MPM_SIMULATOR_H_
#include <memory>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <Setting.h>
#include "../Particle/ParticleDomain.cuh"
#include <Simulation/Model/Model.cuh>
#include <Simulation/Grid/GridDomain.cuh>
#include <Simulation/DomainTransform/DomainTransformer.cuh>
#include <Simulation/TimeIntegrator/TimeIntegrator.cuh>
#include <Partio.h>
#include <MnBase/Math/Probability/Probability.h>
namespace mn {
class Benchmarks;
class MPMSimulator {
  friend class Benchmarks;
  friend class SimulatorBuilder;

 public:
  auto &refModel(int idx = 0) {
    return h_models[idx];
  }
  auto &refGeometryPtr(int idx = 0) {
    return refModel(idx).refGeometryPtr();
  }
  auto &refMaterialPtr(int idx = 0) {
    return refModel(idx).refMaterialDynamicsPtr();
  }
  template <int MaterialType>
  auto getMaterialPtr(int idx = 0);
  auto &refTransformerPtr() {
    return _pTrans;
  }
  auto &refGridPtr() {
    return _pGrid;
  }
  auto &refTimeIntegratorPtr() {
    return _pTimeIntegrator;
  }
  template <int IntegratorType>
  auto getTimeIntegratorPtr();
  MPMSimulator();
  ~MPMSimulator();
  T timeAtFrame(const int frame);
  T computeDt(const T cur, const T next);
  void simulateToFrame(const int frame);
  void simulateToTargetTime(const T targetTime);
  void advanceOneStepExplicitTimeIntegration(const T dt);
  void writePartio(const std::string &filename);
  void buildModel(std::vector<T> &hmass,
                  std::vector<std::array<T, Dim>> &hpos,
                  std::vector<std::array<T, Dim>> &hvel,
                  const int materialType,
                  const T youngsModulus,
                  const T poissonRatio,
                  const T density,
                  const T volume,
                  int idx = 0);
  void buildGrid(const int width,
                 const int height,
                 const int depth,
                 const T dc);
  void buildTransformer(const int gridVolume);
  void buildIntegrator(const int integratorType,
                       const int transferScheme,
                       const T dtDefault);

 private:
  std::string _fileTitle;
  T _dtDefault;
  T _frameRate;
  int _numParticle;  ///< consistent with _pPars
  int _currentFrame;
  T _currentTime;
  std::vector<int> h_indices;
  std::vector<std::array<T, Dim>> h_pos;
  std::vector<std::array<T, Dim>> h_vel;
  std::vector<T> h_mass;
  std::vector<T> h_temp;
  std::vector<T> h_pic_temp;
  std::vector<uint64_t> h_masks;
  /// representation
  std::vector<Model> h_models;
  std::unique_ptr<SPGrid> _pGrid;
  std::unique_ptr<DomainTransformer<TEST_STRUCT<T>>> _pTrans;
  std::unique_ptr<MPMTimeIntegrator> _pTimeIntegrator;
  /// MPM model
  T *d_maxVelSquared;
  T *d_normSquared;
  double *d_innerProduct;
  T *d_tmpMatrix;
  uint64_t *d_masks;
  int _tableSize;
  uint64_t *d_keyTable;  ///< offset (morton code)
  int *d_valTable;       ///< sparse page id
  T *d_memTrunk;
};
}  // namespace mn
#endif
