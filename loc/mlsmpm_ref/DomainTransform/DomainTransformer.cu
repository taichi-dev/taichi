#include "DomainTransformer.cuh"
#include <System/CudaDevice/CudaHostUtils.hpp>
#include <System/CudaDevice/CudaKernelLauncher.cu>
#include <System/Log/Logger.hpp>
#include <Setting.h>
namespace mn {
template <>
void DomainTransformer<TEST_STRUCT<T>>::establishParticleGridMapping() {
  // in order to calc cell num
  checkCudaErrors(cudaMemset(d_marks, 0, sizeof(int) * _numParticle));
  configuredLaunch({"MarkCellBoundary", _numParticle}, markCellBoundary,
                   _numParticle, d_offsets, (d_marks));
  Logger::tick<TimerType::GPU>();
  checkThrustErrors(thrust::inclusive_scan(getDevicePtr(d_marks),
                                           getDevicePtr(d_marks) + _numParticle,
                                           getDevicePtr(d_particle2cell)));
  Logger::tock<TimerType::GPU>("Inclusive scan cell boundary marks");
  checkCudaErrors(cudaMemcpy(&_numCell, (d_particle2cell) + _numParticle - 1,
                             sizeof(int32_t), cudaMemcpyDeviceToHost));
  configuredLaunch({"MarkBlockOffset", _numParticle}, markBlockOffset,
                   _numParticle, (d_particle2cell), (d_cell2particle));
  // page-based
  checkCudaErrors(cudaMemset((d_marks), 0, sizeof(int) * _numParticle));
  configuredLaunch({"MarkPageBoundary", _numParticle}, markPageBoundary,
                   _numParticle, d_offsets, (d_marks));
  Logger::tick<TimerType::GPU>();
  checkThrustErrors(thrust::inclusive_scan(getDevicePtr(d_marks),
                                           getDevicePtr(d_marks) + _numParticle,
                                           getDevicePtr(d_particle2page)));
  Logger::tock<TimerType::GPU>("Inclusive scan page boundary marks");
  checkCudaErrors(cudaMemcpy(&_numPage, (d_particle2page) + _numParticle - 1,
                             sizeof(int32_t), cudaMemcpyDeviceToHost));
  configuredLaunch({"MarkBlockOffset", _numParticle}, markBlockOffset,
                   _numParticle, (d_particle2page), (d_page2particle));
  // page-based
  printf("num particle: %d\tnum page: %d num cell: %d\n", _numParticle,
         _numPage, _numCell);
}
template <>
void DomainTransformer<TEST_STRUCT<T>>::establishHashmap() {
  checkCudaErrors(cudaMemset(d_keyTable, 0xff, sizeof(uint64_t) * _tableSize));
  checkCudaErrors(cudaMemset(d_valTable, 0xff, sizeof(int) * _tableSize));
  printf("xxxxx table size is %d\n", _tableSize);
  recordLaunch(
      std::string("EstablishHashmapFromPage"), (int)(_numPage + 511) / 512, 512,
      (size_t)0, buildHashMapFromPage, _numPage, _tableSize, d_masks, d_offsets,
      (const int *)(d_page2particle), d_keyTable, d_valTable, d_pageOffset);
  checkCudaErrors(
      cudaMemcpy(d_totalPage, &_numPage, sizeof(int), cudaMemcpyHostToDevice));
  recordLaunch(std::string("SupplementAdjacentPages"),
               (int)(_numPage + 511) / 512, 512, (size_t)0,
               supplementAdjacentPages, _numPage, _tableSize, d_masks,
               d_offsets, (const uint64_t *)d_neighborOffsets,
               (const int *)(d_page2particle), d_keyTable, d_valTable,
               d_totalPage, d_pageOffset);
  recordLaunch(
      std::string("EstablishPageTopology"), (int)(_numPage + 511) / 512, 512,
      (size_t)0, establishPageTopology, _numPage, _tableSize, d_masks,
      d_offsets, (const uint64_t *)d_neighborOffsets,
      (const int *)(d_page2particle), d_keyTable, d_valTable, d_adjPage);
  checkCudaErrors(cudaMemcpy(&_numTotalPage, d_totalPage, sizeof(int32_t),
                             cudaMemcpyDeviceToHost));
}
template <>
void DomainTransformer<TEST_STRUCT<T>>::buildTargetPage() {
  configuredLaunch({"MarkPageSize", _numPage}, markPageSize, _numPage,
                   (const int32_t *)d_page2particle, d_marks);
  Logger::tick<TimerType::GPU>();
  checkThrustErrors(thrust::exclusive_scan(getDevicePtr(d_marks),
                                           getDevicePtr(d_marks) + _numPage + 1,
                                           getDevicePtr(d_virtualPageOffset)));
  Logger::tock<TimerType::GPU>("Exclusive scan page capacity marks");
  checkCudaErrors(cudaMemcpy(&_numVirtualPage, d_virtualPageOffset + _numPage,
                             sizeof(int32_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemset(d_marks, 0, sizeof(int32_t) * _numVirtualPage));
  configuredLaunch({"MarkVirtualPageOffset", _numPage}, markVirtualPageOffset,
                   _numPage, (const int32_t *)d_virtualPageOffset, d_marks);
  Logger::tick<TimerType::GPU>();
  checkThrustErrors(thrust::inclusive_scan(
      getDevicePtr(d_marks), getDevicePtr(d_marks) + _numVirtualPage,
      getDevicePtr(d_targetPage)));
  Logger::tock<TimerType::GPU>("Inclusive scan page boundary marks");
}
}  // namespace mn
