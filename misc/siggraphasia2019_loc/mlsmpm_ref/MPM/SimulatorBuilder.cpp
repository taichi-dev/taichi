#include "SimulatorBuilder.h"
#include "ModelInitSchemes.hpp"
#include <System/Log/Logger.hpp>
#include <SPGrid_Allocator.h>
namespace mn {
void matrixVectorMultiplication(const T *x, const T *v, T *result) {
  result[0] = x[0] * v[0] + x[3] * v[1] + x[6] * v[2];
  result[1] = x[1] * v[0] + x[4] * v[1] + x[7] * v[2];
  result[2] = x[2] * v[0] + x[5] * v[1] + x[8] * v[2];
}
std::string getMaterialName(int opt) {
  if (opt == 0)
    return "neohookean";
  else if (opt == 1)
    return "fixedcorotated";
  else if (opt == 2)
    return "newSand";
  else if (opt == 3)
    return "snow";
  else
    return "unknownmaterial";
}
std::string getTransferScheme(int opt) {
  if (opt == 0)
    return "flip";
  else if (opt == 1)
    return "apic";
  else if (opt == 2)
    return "mls";
  else
    return "unknown";
}
bool SimulatorBuilder::build(MPMSimulator &simulator, int opt) {
  simulator.h_masks.resize(Dim);
  // hardcode mask
  using namespace SPGrid;
  using T_STRUCT = TEST_STRUCT<T>;
  using SPG_Allocator = SPGrid_Allocator<T_STRUCT, Dim>;
  using T_MASK = typename SPG_Allocator::Array<T>::mask;
  simulator.h_masks[0] = T_MASK::xmask;
  simulator.h_masks[1] = T_MASK::ymask;
  simulator.h_masks[2] = T_MASK::zmask;
  // configuration
  T totalMass, volume, pMass;
  int materialType, geometryType;
  T youngsModulus, poissonRatio, density, totalVolume;
  int integratorType, transferScheme;
  T dtDefault;
  std::string prefix;
  if (opt == 0) {
    // material
    materialType = 1;  // 0 neohookean 1 fixed corotated
    youngsModulus = 5e3;
    poissonRatio = 0.4;
    density = 1000.f;
    totalVolume = 1.f;
    geometryType = 0;
    simulator._tableSize = 21000000;
    // position
    simulator.h_pos = Initialize_Data<T, 0>();
    simulator._numParticle = simulator.h_pos.size();
    totalVolume = 0.75f * 0.75f * 0.75f;
    // velocity
    simulator.h_vel.resize(simulator._numParticle);
    std::fill(simulator.h_vel.begin(), simulator.h_vel.end(),
              std::array<T, Dim>{0, 0, 0});
    // integrator
    transferScheme = TRANSFER_SCHEME;  // 0 flip 1 apic 2 mls
    dtDefault = integratorType == 0 ? 1e-4 : 1e-3;
    // file
    prefix = "benchmark_cube_test_";
  } else if (opt == 1) {
    // material
    materialType = 1;  // 0 neohookean 1 fixed corotated
    youngsModulus = 5e3;
    poissonRatio = 0.4;
    density = 1000.f;
    totalVolume = 1.f;
    geometryType = 7;
    simulator._tableSize = 100000;
    // position
    {
      srand(0);
      int center[3] = {N / 2, N / 2, N / 2};
      int res[3] = {5 * N / 6, 5 * N / 6, 5 * N / 6};
      int minCorner[3];
      for (int i = 0; i < 3; ++i)
        minCorner[i] = center[i] - .5 * res[i];
      simulator.h_pos = Initialize_Data<T, 7>();
      int sizeOfOneCopy = simulator.h_pos.size();
      T stride = (N - center[0] * 2) * 0.3333 * dx;
      for (int i = 0; i < 1; ++i)
        for (int j = 0; j < 1; ++j) {
          if (i == 0 && j == 0)
            continue;
          T theta = (T)(10. / 180.) * 3.1415926f;
          T rotation[9] = {std::cos(theta),
                           -std::sin(theta),
                           0,
                           std::sin(theta),
                           std::cos(theta),
                           0,
                           0,
                           0,
                           1};
          for (int p = 0; p < sizeOfOneCopy; ++p) {
            std::array<T, 3> &pos = simulator.h_pos[p];
            // move pos to center
            T diff[3];
            for (int i = 0; i < 3; ++i)
              diff[i] = pos[i] - center[i] * dx;
            matrixVectorMultiplication(rotation, diff, pos.data());
            for (int i = 0; i < 3; ++i)
              pos[i] = pos[i] + center[i] * dx;
          }
        }
      totalVolume = res[0] * res[1] * res[2] * dx * dx * dx;
    }
    simulator._numParticle = simulator.h_pos.size();
    // velocity
    simulator.h_vel.resize(simulator._numParticle);
    std::fill(simulator.h_vel.begin(), simulator.h_vel.end(),
              std::array<T, Dim>{0, 0, 0});
    int numPartParticle = 0;
    for (int i = 0; i < simulator._numParticle; i++)
      if (simulator.h_pos[i][2] > 0.5f)
        numPartParticle++;
    {
      std::cout << "The Number of Particles for the first part"
                << numPartParticle << std::endl;
      std::vector<std::array<T, Dim>> tmp = simulator.h_pos;
      int cubeIndex = 0;
      int baseIndex = numPartParticle;
      for (int i = 0; i < simulator._numParticle; i++)
        if (tmp[i][2] > 0.5) {
          simulator.h_pos[cubeIndex] = tmp[i];
          simulator.h_vel[cubeIndex++][2] = -0.5;
        } else {
          simulator.h_pos[baseIndex] = tmp[i];
          simulator.h_vel[baseIndex++][2] = +0.5;
        }
      tmp.clear();
    }
    // integrator
    dtDefault = integratorType == 0 ? 1e-4 : 1e-3;
    // file
    prefix = "benchmark_two_dragons_gravity_";
  } else
    return false;
  //////////////////////////////////////////////////////////////////////////
  // indices
  simulator.h_indices.resize(simulator._numParticle);
  for (int i = 0; i < simulator._numParticle; ++i)
    simulator.h_indices[i] = i;
  int numPartParticle = 0;
  totalMass = density * totalVolume;
  volume = totalVolume / simulator._numParticle;
  pMass = totalMass / simulator._numParticle;
  //////////////////////////////////////////////////////////////////////////
  // mass
  simulator.h_mass.resize(simulator._numParticle);
  for (auto &mass : simulator.h_mass)
    mass = pMass;
  //////////////////////////////////////////////////////////////////////////
  // file title
  char str[500];
  sprintf(str, "%s%s_initialize(%d)_%s_par[%d]_grid[%d]_%s_%e_", prefix.data(),
          getTransferScheme(transferScheme).data(), opt,
          getMaterialName(materialType).data(), simulator._numParticle, N,
          integratorType ? "imp" : "exp", dtDefault);
  simulator._fileTitle = std::string(str);
  ///
  reportMemory("before simulator_allocation");
  checkCudaErrors(
      cudaMalloc((void **)&simulator.d_memTrunk,
                 sizeof(T) * simulator._numParticle * (21)));  // tmp(21)
  checkCudaErrors(cudaMalloc((void **)&simulator.d_maxVelSquared, sizeof(T)));
  checkCudaErrors(
      cudaMalloc((void **)&simulator.d_masks, sizeof(uint64_t) * Dim));
  checkCudaErrors(cudaMemcpy(simulator.d_masks, simulator.h_masks.data(),
                             sizeof(uint64_t) * Dim, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&simulator.d_keyTable,
                             sizeof(uint64_t) * simulator._tableSize));
  checkCudaErrors(cudaMalloc((void **)&simulator.d_valTable,
                             sizeof(int) * simulator._tableSize));
  reportMemory("after simulator_allocation");
  /// components
  printf("volume is %f\n", volume);
  if (materialType == 0 || materialType == 1)
    simulator.buildModel(simulator.h_mass, simulator.h_pos, simulator.h_vel,
                         materialType, youngsModulus, poissonRatio, density,
                         volume, 0);
  simulator.buildGrid(N, N, N, dx);
  simulator.buildTransformer(N * N * N);
  simulator.buildIntegrator(integratorType, transferScheme, dtDefault);
#ifdef SAVE_DATA
  simulator.writePartio(simulator._fileTitle + std::to_string(0) + ".bgeo");
#endif  // SAVE_DATA
  Logger::blankLine<TimerType::GPU>();
  return true;
}
}  // namespace mn