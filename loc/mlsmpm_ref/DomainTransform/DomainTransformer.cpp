#include "DomainTransformer.cuh"
#include <Setting.h>
#include <MnBase/Meta/AllocMeta.h>
#include <MnBase/AggregatedAttribs.h>
#include <SPGrid_Allocator.h>
namespace mn {
template <>
DomainTransformer<TEST_STRUCT<T>>::DomainTransformer(const int numParticle,
                                                     const int gridVolume)
    : _numParticle(numParticle), _gridVolume(gridVolume) {
  reportMemory("before domain_transformer_allocation");
  _attribs = cuda_allocs<TransformerAttribs>(numParticle + 2);
  d_marks = (int32_t *)_attribs[(int)TransformerAttribIndex::MARK];
  d_particle2cell = (int32_t *)_attribs[(int)TransformerAttribIndex::PAR2CELL];
  d_cell2particle = (int32_t *)_attribs[(int)TransformerAttribIndex::CELL2PAR];
  d_particle2page = (int32_t *)_attribs[(int)TransformerAttribIndex::PAR2PAGE];
  d_page2particle = (int32_t *)_attribs[(int)TransformerAttribIndex::PAGE2PAR];
  d_virtualPageOffset =
      (int32_t *)_attribs[(int)TransformerAttribIndex::VIR_PAGE_OFFSET];
  d_targetPage = (int32_t *)_attribs[(int)TransformerAttribIndex::TAR_PAGE];
  checkCudaErrors(cudaMalloc((void **)&d_totalPage, sizeof(int)));
  for (auto &adjpage : hd_adjPage)
    checkCudaErrors(
        cudaMalloc((void **)&adjpage,
                   sizeof(int) * (int)(gridVolume * MEMORY_SCALE / 4 / 4 / 4)));
  checkCudaErrors(cudaMalloc((void **)&d_adjPage, sizeof(int *) * 7));
  checkCudaErrors(cudaMemcpy(d_adjPage, hd_adjPage, sizeof(int *) * 7,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(
      (void **)&d_pageOffset,
      sizeof(uint64_t) * (int)(gridVolume * MEMORY_SCALE / 4 / 4 / 4)));
  uint64_t h_neighborOffsets[8];
  using namespace SPGrid;
  using T_STRUCT = TEST_STRUCT<T>;
  using SPG_Allocator = SPGrid_Allocator<T_STRUCT, Dim>;
  using T_MASK = typename SPG_Allocator::Array<T>::mask;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        h_neighborOffsets[i * 4 + j * 2 + k] =
            T_MASK::Linear_Offset(i * 4, j * 4, k * 4);
  checkCudaErrors(
      cudaMalloc((void **)&d_neighborOffsets, sizeof(uint64_t) * 7));
  checkCudaErrors(cudaMemcpy(d_neighborOffsets, h_neighborOffsets + 1,
                             sizeof(uint64_t) * 7, cudaMemcpyHostToDevice));
  // 3 - 0 1 1
  // 4 - 1 0 0
  // 5 - 1 0 1
  // 6 - 1 1 0
  reportMemory("after domain_transformer_allocation");
}
template <>
void DomainTransformer<TEST_STRUCT<T>>::initialize(const uint64_t *dMasks,
                                                   const uint64_t *dOffsets,
                                                   const int tableSize,
                                                   uint64_t *keyTable,
                                                   int *valTable) {
  d_masks = dMasks;
  d_offsets = dOffsets;
  _tableSize = tableSize;
  d_keyTable = keyTable;
  d_valTable = valTable;
}
template <>
DomainTransformer<TEST_STRUCT<T>>::~DomainTransformer() {
  checkCudaErrors(cudaFree(d_pageOffset));
  checkCudaErrors(cudaFree(d_neighborOffsets));
  checkCudaErrors(cudaFree(d_adjPage));
  for (auto &adjpage : hd_adjPage)
    checkCudaErrors(cudaFree(adjpage));
  checkCudaErrors(cudaFree(d_totalPage));
}
template <>
void DomainTransformer<TEST_STRUCT<T>>::rebuild() {
  // clean before computing
  checkCudaErrors(cudaMemset(
      (void *)d_pageOffset, 0,
      sizeof(uint64_t) * (int)(_gridVolume * MEMORY_SCALE / 4 / 4 / 4)));
  for (auto &adjpage : hd_adjPage)
    checkCudaErrors(cudaMemset(
        (void *)adjpage, 0,
        sizeof(int) * (int)(_gridVolume * MEMORY_SCALE / 4 / 4 / 4)));
  //
  establishParticleGridMapping();
  establishHashmap();
  checkCudaErrors(cudaMemcpy(&_numTotalPage, d_totalPage, sizeof(int32_t),
                             cudaMemcpyDeviceToHost));
  buildTargetPage();
  printf("_numTotalPage is %d \n", _numTotalPage);
}
}  // namespace mn
