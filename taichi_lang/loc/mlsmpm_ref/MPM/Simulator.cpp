#include "Simulator.h"
#include "../Particle/ParticleDomain.cuh"
#include "../Grid/GridDomain.cuh"
#include "../DomainTransform/DomainTransformer.cuh"
#include <System/Log/Logger.hpp>
#include <stdio.h>
#include <stdlib.h>
namespace mn {
MPMSimulator::MPMSimulator()
    : _frameRate(48.f), _currentFrame(0), _currentTime(0) {
}
MPMSimulator::~MPMSimulator() {
  checkCudaErrors(cudaFree(d_keyTable));
  checkCudaErrors(cudaFree(d_valTable));
  checkCudaErrors(cudaFree(d_masks));
  checkCudaErrors(cudaFree(d_maxVelSquared));
  checkCudaErrors(cudaFree(d_memTrunk));
}
void MPMSimulator::buildModel(std::vector<T> &hmass,
                              std::vector<std::array<T, Dim>> &hpos,
                              std::vector<std::array<T, Dim>> &hvel,
                              const int materialType,
                              const T youngsModulus,
                              const T poissonRatio,
                              const T density,
                              const T volume,
                              int idx) {
  {
    auto geometry = std::make_unique<Particles>(_numParticle, _tableSize);
    auto material = std::make_unique<ElasticMaterialDynamics>(
        materialType, youngsModulus, poissonRatio, density, volume);
    h_models.emplace_back(
        std::make_unique<Model>(std::move(geometry), std::move(material)));
    printf("\n\tFinish Particles initializing\n");
  }
  auto &geometry = refGeometryPtr(idx);
  geometry->initialize(hmass.data(), (const T *)hpos.data(),
                       (const T *)hvel.data(), d_masks, h_masks.data(),
                       d_keyTable, d_valTable, d_memTrunk);
}
void MPMSimulator::buildGrid(const int width,
                             const int height,
                             const int depth,
                             const T dc) {
  auto &grid = refGridPtr();
  grid = std::make_unique<SPGrid>(width, height, depth, dc);
  grid->initialize(d_masks, h_masks.data());
  printf("\n\tFinish SPGrid initializing\n");
}
void MPMSimulator::buildTransformer(const int gridVolume) {
  auto &transformer = refTransformerPtr();
  transformer = std::make_unique<DomainTransformer<TEST_STRUCT<T>>>(
      _numParticle, gridVolume);
  transformer->initialize(d_masks, refGeometryPtr(0)->d_offsets, _tableSize,
                          d_keyTable, d_valTable);
  printf("\n\tFinish Transformer initializing\n");
  Logger::blankLine<TimerType::GPU>();
}
void MPMSimulator::writePartio(const std::string &filename) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  Partio::ParticleAttribute posH =
      parts->addAttribute("position", Partio::VECTOR, 3);
  Partio::ParticleAttribute velH =
      parts->addAttribute("velocity", Partio::VECTOR, 3);
  Partio::ParticleAttribute indexH =
      parts->addAttribute("index", Partio::INT, 1);
  Partio::ParticleAttribute typeH = parts->addAttribute("type", Partio::INT, 1);
  for (unsigned int i = 0; i < h_pos.size(); ++i) {
    int idx = parts->addParticle();
    float *p = parts->dataWrite<float>(posH, idx);
    float *v = parts->dataWrite<float>(velH, idx);
    int *index = parts->dataWrite<int>(indexH, idx);
    int *type = parts->dataWrite<int>(typeH, idx);
    for (int k = 0; k < 3; k++)
      p[k] = h_pos[i][k];
    for (int k = 0; k < 3; k++)
      v[k] = h_vel[i][k];
    index[0] = h_indices[i];
    type[0] = 1;
  }
  Partio::write(filename.c_str(), *parts);
  parts->release();
}
}  // namespace mn
