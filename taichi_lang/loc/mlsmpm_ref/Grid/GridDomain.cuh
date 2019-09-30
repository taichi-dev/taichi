#ifndef __GRID_DOMAIN_H_
#define __GRID_DOMAIN_H_
#include <MnBase/AggregatedAttribs.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <System/CudaDevice/CudaHostUtils.hpp>
#include <helper_cuda.h>
#include <stdint.h>
#include <Setting.h>
#include "GridKernels.cuh"
namespace mn {
// offset mass ordermass posx posy posz ordposx ordposy ordposz ordvelx ordvely
// ordvelz indexx indexy indexz offsetx offsety offsetz
// enum GridAttribIndex { NUM_ATTRIBS};
// using ParticleAttribs = std::tuple<uint64_t, T, T, T, T, T, T, T, T, T, T, T,
// int, int, int, uint64_t, uint64_t, uint64_t>;
class
    SPGrid {  //: public
              // AttribConnector<static_cast<int>(ParticleAttribIndex::NUM_ATTRIBS),
              // 1> {
 public:
  friend class MPMSimulator;
  SPGrid() = delete;
  SPGrid(int w, int h, int d, T dc);
  void initialize(const uint64_t *dMasks, const uint64_t *hMasks);
  ~SPGrid();
  void clear();
  void clearImplicitGrid();

 public:
  /// input
  int _width, _height, _depth;
  T _memoryScale;
  T _dc;
  const uint64_t *d_masks;
  const uint64_t *h_masks;
  // grid data
  T *hd_channels[15];
  T *d_grid;
  T **d_channels;
  unsigned *d_flags;
  // implicit solve
  T *d_grid_implicit;
  unsigned *d_implicit_flags;
  T *hd_implicit_x[3];
  T **d_implicit_x;
  T *hd_implicit_p[3];
  T **d_implicit_p;
  T *hd_implicit_ap[3];
  T **d_implicit_ap;
  T *hd_implicit_r[3];
  T **d_implicit_r;
  T *hd_implicit_ar[3];
  T **d_implicit_ar;
};
}  // namespace mn
#endif
