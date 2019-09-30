#include "Simulator.h"
#include "SimulatorKernels.cuh"
#include <MnBase/Math/Matrix/MatrixKernels.cuh>
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <System/CudaDevice/CudaKernelLauncher.cu>
#include <cmath>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}
namespace mn {
template <>
auto MPMSimulator::getMaterialPtr<0>(int idx) {
  return (ElasticMaterialDynamics *)refMaterialPtr(idx).get();
}
// explicit elasticity
template <>
auto MPMSimulator::getTimeIntegratorPtr<0>() {
  return (ExplicitTimeIntegrator *)refTimeIntegratorPtr().get();
}
// implicit elasticity
template <>
auto MPMSimulator::getTimeIntegratorPtr<1>() {
  return (ImplicitTimeIntegrator *)refTimeIntegratorPtr().get();
}
void MPMSimulator::buildIntegrator(const int integratorType,
                                   const int transferScheme,
                                   const T dtDefault) {
  if (integratorType == 1)
    _pTimeIntegrator = std::make_unique<ImplicitTimeIntegrator>(
        transferScheme, _numParticle, d_memTrunk);
  else
    _pTimeIntegrator = std::make_unique<ExplicitTimeIntegrator>(
        transferScheme, _numParticle, d_memTrunk);
  printf("\n4\tFinish Integrator initializing\n");
  _dtDefault = dtDefault;
  configuredLaunch({"InitMatrix", refGeometryPtr(0)->_numParticle}, initMatrix,
                   refGeometryPtr(0)->_numParticle, refGeometryPtr(0)->d_F);
  if (transferScheme)
    checkCudaErrors(
        cudaMemset(refGeometryPtr(0)->d_B, 0,
                   sizeof(T) * 9 * refGeometryPtr(0)->_numParticle));
  Logger::blankLine<TimerType::GPU>();
}
T MPMSimulator::timeAtFrame(const int frame) {
  // should be above 0
  return (T)frame / (T)_frameRate;
}
T MPMSimulator::computeDt(const T cur, const T next) {
  T dt = _dtDefault, maxVel = 0;
  checkCudaErrors(cudaMemcpy(d_maxVelSquared, (void *)&maxVel, sizeof(T),
                             cudaMemcpyHostToDevice));
  configuredLaunch({"CalcMaxVel", refGeometryPtr(0)->_numParticle}, calcMaxVel,
                   refGeometryPtr(0)->_numParticle,
                   (const T **)refGeometryPtr(0)->d_orderedVel,
                   d_maxVelSquared);
  checkCudaErrors(cudaMemcpy((void *)&maxVel, d_maxVelSquared, sizeof(T),
                             cudaMemcpyDeviceToHost));
  maxVel = sqrt(maxVel);
  printf("maxVel is %f ", maxVel);
  if (maxVel > 0 && (maxVel = dx * .3f / maxVel) < _dtDefault)
    dt = maxVel;
  /*maxVel = next - cur;*/
  if (cur + dt >= next)
    dt = next - cur;
  else if (cur + 2 * dt >= next && (maxVel = (next - cur) * 0.51) < dt)
    dt = maxVel;
  printf(" dt is %f curr is %f and next is %f \n", dt, cur, next);
  return dt;
}
void MPMSimulator::simulateToFrame(const int frame) {
  printf("Simulate from Frame %d to Frame %d\n", _currentFrame, frame);
  while (_currentFrame < frame) {
    printf("\nSimulating frame %d\n", ++_currentFrame);
    simulateToTargetTime(timeAtFrame(_currentFrame));
    printf("END Frame %d\n", _currentFrame);
    Logger::recordFrame<TimerType::GPU>();
    refGeometryPtr(0)->retrieveAttribs(h_pos, h_vel, h_indices);
#ifdef SAVE_DATA
    writePartio(_fileTitle + std::to_string(_currentFrame) + ".bgeo");
#endif  // SAVE_DATA
  }
}
void MPMSimulator::simulateToTargetTime(const T targetTime) {
  printf("\n\t\tSimulate from Time %.6f to Time %.6f\n", _currentTime,
         targetTime);
  auto t = get_time();
  int substep = 1;
  int num_substeps = 0;
  do {
    // T dt = computeDt(_currentTime, targetTime);
    // printf("\n\t\t\tSubstep %d [%.6f, %.6f] dt: %.6f\n", substep++,
    // _currentTime, _currentTime + dt, dt);
    T dt = 0.000100;  // fixed dt
    advanceOneStepExplicitTimeIntegration(dt);
    _currentTime += dt;
    num_substeps++;
  } while (_currentTime < targetTime);
  std::cout << " ##### average substep time : "
            << 1000 * (get_time() - t) / num_substeps << "ms"
            << " num substeps: " << num_substeps << std::endl;
  // exit(0);  nvprof --print-gpu-trace ./mpm
}
void MPMSimulator::advanceOneStepExplicitTimeIntegration(const T dt) {
  getTimeIntegratorPtr<MPM_SIM_TYPE>()->integrate(dt, refModel(0), refGridPtr(),
                                                  refTransformerPtr());
}
}  // namespace mn
