#ifndef __DOMAIN_TRANSFORMER_H_
#define __DOMAIN_TRANSFORMER_H_
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <System/CudaDevice/CudaHostUtils.hpp>
#include <helper_cuda.h>
#include <stdint.h>
#include <Setting.h>
#include "DomainTransformKernels.cuh"
#include "../Particle/ParticleDomain.cuh"
#include "../Grid/GridDomain.cuh"
namespace mn {
enum class TransformerAttribIndex {
  MARK = 0,
  PAR2CELL,
  CELL2PAR,
  PAR2PAGE,
  PAGE2PAR,
  VIR_PAGE_OFFSET,
  TAR_PAGE,
  NUM_ATTRIBS
};
using TransformerAttribs =
    std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>;
/// for particle-based sparse page construction demonstration
template <typename T_STRUCT>
class DomainTransformer
    : public AttribConnector<static_cast<int>(
                                 TransformerAttribIndex::NUM_ATTRIBS),
                             1> {
 public:
  friend class MPMSimulator;
  void establishParticleGridMapping();
  void establishHashmap();
  void buildTargetPage();
  DomainTransformer() = delete;
  DomainTransformer(const int numParticle, const int gridVolume);
  void initialize(const uint64_t *dMasks,
                  const uint64_t *dOffsets,
                  const int tableSize,
                  uint64_t *keyTable,
                  int *valTable);
  ~DomainTransformer();
  void rebuild();

 public:
  /// input
  int _numParticle;
  int _gridVolume;
  int _tableSize;
  const uint64_t *d_masks;
  const uint64_t *d_offsets;
  uint64_t *d_keyTable;  ///< offset (morton code)
  int *d_valTable;       ///< sparse page id
                         /// hash
  int *d_totalPage;
  int *hd_adjPage[7];
  int **d_adjPage;
  uint64_t *d_pageOffset;
  int _numPage, _numCell;
  int _numTotalPage, _numVirtualPage;
  uint64_t *d_neighborOffsets;
  /// mapping
  int32_t *d_marks;
  int32_t *d_particle2page;  ///< particle no. -> cell no.
  int32_t *d_page2particle;  ///< cell no. -> particle offset
  int32_t *d_particle2cell;
  int32_t *d_cell2particle;
  // int32_t*          d_pageSize;       ///< number of particles (or blocks
  // needed) per page, replaced by d_marks
  int32_t *d_virtualPageOffset;
  int32_t *d_targetPage;  ///< virtual page no. -> actual (target) page no.
};
}  // namespace mn
#endif
