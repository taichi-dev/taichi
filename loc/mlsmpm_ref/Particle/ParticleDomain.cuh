#ifndef __PARTICLE_DOMAIN_H_
#define __PARTICLE_DOMAIN_H_
#include <MnBase/AggregatedAttribs.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <System/CudaDevice/CudaHostUtils.hpp>
#include <helper_cuda.h>
#include <stdint.h>
#include <Setting.h>
#include "ParticleKernels.cuh"
namespace mn {
// offset mass ordermass posx posy posz ordposx ordposy ordposz ordvelx ordvely
// ordvelz indexx indexy indexz offsetx offsety offsetz
enum class ParticleAttribIndex {
  OFFSET = 0,
  MASS,
  ORD_MASS,
  POSX,
  POSY,
  POSZ,
  ORD_POSX,
  ORD_POSY,
  ORD_POSZ,
  ORD_VELX,
  ORD_VELY,
  ORD_VELZ,
  INDEXX,
  INDEXY,
  INDEXZ,
  NUM_ATTRIBS
};
using ParticleAttribs =
    std::tuple<uint64_t, T, T, T, T, T, T, T, T, T, T, T, int, int, int>;
/// for particle-based sparse page construction demonstration
class Particles
    : public AttribConnector<static_cast<int>(ParticleAttribIndex::NUM_ATTRIBS),
                             1> {
 public:
  friend class MPMSimulator;
  Particles() = delete;
  Particles(int numParticle, int tableSize);
  ~Particles();
  void initialize(const T *h_mass,
                  const T *h_pos,
                  const T *h_vel,
                  const uint64_t *_dmasks,
                  const uint64_t *_hmasks,
                  uint64_t *keyTable,
                  int *valTable,
                  void *_memTrunk);
  void sort_by_offsets();
  void reorder();
  void retrieveAttribs(std::vector<std::array<T, Dim>> &h_pos,
                       std::vector<std::array<T, Dim>> &h_vel,
                       std::vector<int> &h_indices);

 public:
  /// input
  int _numParticle;
  const uint64_t *d_masks;
  const uint64_t *h_masks;
  float3 *d_min;
  float3 _min;
  /// attribs
  uint64_t *d_offsets;
  T *d_mass;
  T *d_orderedMass;  ///< initial data
  T *hd_pos[Dim];
  T *hd_orderedPos[Dim];
  T *hd_orderedVel[Dim];
  int *hd_smallestNodeIndex[Dim];
  T **d_pos;
  T **d_orderedPos;  ///< initial data
  T **d_orderedVel;  ///< initial data
  int **d_smallestNodeIndex;
  /// auxiliary
  T *d_tmp;
  int *d_indices;
  int *d_indexTrans;
  int _numBucket;
  int *d_numBucket;
  int *d_particle2bucket;
  int *d_bucketSizes;
  int *d_bucketOffsets;
  uint8_t *d_cellKeys;
  int _tableSize;
  uint64_t *d_keyTable;  ///< offset (morton code)
  int *d_valTable;       ///< sparse page id
  T *d_F;
  T *d_B;
};
}  // namespace mn
#endif
