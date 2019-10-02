#include "GridDomain.cuh"
#include <SPGrid_Allocator.h>
namespace mn {
SPGrid::SPGrid(int weight, int height, int depth, T dc)
    : _width(weight), _height(height), _depth(depth), _dc(dc) {
  using namespace SPGrid;
  using T_STRUCT = TEST_STRUCT<T>;
  using SPG_Allocator = SPGrid_Allocator<T_STRUCT, Dim>;
  using T_MASK = typename SPG_Allocator::Array<T>::mask;
  _memoryScale = MEMORY_SCALE;
  reportMemory("before sparse_grid_allocation");
  checkCudaErrors(
      cudaMalloc((void **)&d_grid,
                 sizeof(struct TEST_STRUCT<T>) * _width * _height * _depth *
                     _memoryScale));  // more secured way should be used
  int tmp =
      sizeof(struct TEST_STRUCT<T>) * _width * _height * _depth * _memoryScale;
  checkCudaErrors(cudaMalloc((void **)&d_grid_implicit, tmp));
  checkCudaErrors(cudaMalloc((void **)&d_channels, sizeof(T *) * 15));
  checkCudaErrors(cudaMalloc((void **)&d_implicit_x, sizeof(T *) * 3));
  checkCudaErrors(cudaMalloc((void **)&d_implicit_p, sizeof(T *) * 3));
  checkCudaErrors(cudaMalloc((void **)&d_implicit_ap, sizeof(T *) * 3));
  checkCudaErrors(cudaMalloc((void **)&d_implicit_r, sizeof(T *) * 3));
  checkCudaErrors(cudaMalloc((void **)&d_implicit_ar, sizeof(T *) * 3));
  printf("size one struct(%d Bytes) total(%d Bytes)\n",
         (int)sizeof(struct TEST_STRUCT<T>), tmp);
  hd_channels[0] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch0) *
                                  T_MASK::elements_per_block));  // mass
  hd_channels[1] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch1) *
                                  T_MASK::elements_per_block));  // vel momentum
  hd_channels[2] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch2) *
                                  T_MASK::elements_per_block));  // vel
  hd_channels[3] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch3) *
                                  T_MASK::elements_per_block));  // vel
  hd_channels[4] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch4) *
                                  T_MASK::elements_per_block));  // force
  hd_channels[5] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch5) *
                                  T_MASK::elements_per_block));  // force
  hd_channels[6] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch6) *
                                  T_MASK::elements_per_block));  // force
  hd_channels[7] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch7) *
                                  T_MASK::elements_per_block));  // vel0
  hd_channels[8] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch8) *
                                  T_MASK::elements_per_block));  // vel0
  hd_channels[9] = reinterpret_cast<T *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch9) *
                                  T_MASK::elements_per_block));  // vel0
  checkCudaErrors(cudaMemcpy(d_channels, hd_channels, sizeof(T *) * 15,
                             cudaMemcpyHostToDevice));
  d_flags = reinterpret_cast<unsigned *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&TEST_STRUCT<T>::flags) *
                                  T_MASK::elements_per_block));
  // implicit solve
  d_implicit_flags = reinterpret_cast<unsigned *>(
      (uint64_t)d_grid_implicit +
      uint64_t(OffsetOfMember(&TEST_STRUCT<T>::flags) *
               T_MASK::elements_per_block));
  hd_implicit_x[0] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch0) *
                                     T_MASK::elements_per_block));
  hd_implicit_x[1] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch1) *
                                     T_MASK::elements_per_block));
  hd_implicit_x[2] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch2) *
                                     T_MASK::elements_per_block));
  checkCudaErrors(cudaMemcpy(d_implicit_x, hd_implicit_x, sizeof(T *) * 3,
                             cudaMemcpyHostToDevice));
  hd_implicit_p[0] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch3) *
                                     T_MASK::elements_per_block));
  hd_implicit_p[1] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch4) *
                                     T_MASK::elements_per_block));
  hd_implicit_p[2] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch5) *
                                     T_MASK::elements_per_block));
  checkCudaErrors(cudaMemcpy(d_implicit_p, hd_implicit_p, sizeof(T *) * 3,
                             cudaMemcpyHostToDevice));
  hd_implicit_ap[0] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch6) *
                                     T_MASK::elements_per_block));
  hd_implicit_ap[1] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch7) *
                                     T_MASK::elements_per_block));
  hd_implicit_ap[2] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch8) *
                                     T_MASK::elements_per_block));
  checkCudaErrors(cudaMemcpy(d_implicit_ap, hd_implicit_ap, sizeof(T *) * 3,
                             cudaMemcpyHostToDevice));
  hd_implicit_r[0] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch9) *
                                     T_MASK::elements_per_block));
  hd_implicit_r[1] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch10) *
                                     T_MASK::elements_per_block));
  hd_implicit_r[2] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch11) *
                                     T_MASK::elements_per_block));
  checkCudaErrors(cudaMemcpy(d_implicit_r, hd_implicit_r, sizeof(T *) * 3,
                             cudaMemcpyHostToDevice));
  hd_implicit_ar[0] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch12) *
                                     T_MASK::elements_per_block));
  hd_implicit_ar[1] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch13) *
                                     T_MASK::elements_per_block));
  hd_implicit_ar[2] =
      reinterpret_cast<T *>((uint64_t)d_grid_implicit +
                            uint64_t(OffsetOfMember(&TEST_STRUCT<T>::ch14) *
                                     T_MASK::elements_per_block));
  checkCudaErrors(cudaMemcpy(d_implicit_ar, hd_implicit_ar, sizeof(T *) * 3,
                             cudaMemcpyHostToDevice));
  reportMemory("after sparse_grid_allocation");
}
void SPGrid::initialize(const uint64_t *dMasks, const uint64_t *hMasks) {
  d_masks = dMasks;
  h_masks = hMasks;
}
SPGrid::~SPGrid() {
  checkCudaErrors(cudaFree(d_grid));
  checkCudaErrors(cudaFree(d_channels));
  checkCudaErrors(cudaFree(d_grid_implicit));
  checkCudaErrors(cudaFree(d_implicit_x));
  checkCudaErrors(cudaFree(d_implicit_p));
  checkCudaErrors(cudaFree(d_implicit_r));
  checkCudaErrors(cudaFree(d_implicit_ap));
  checkCudaErrors(cudaFree(d_implicit_ar));
}
void SPGrid::clear() {
  // checkCudaErrors(cudaMemset(d_grid, 0, sizeof(struct
  // TEST_STRUCT<T>)*_width*_height*_depth*_memoryScale));
  // checkCudaErrors(cudaMemset(d_grid_implicit, 0, sizeof(struct
  // TEST_STRUCT<T>)*_width*_height*_depth*_memoryScale));
}
void SPGrid::clearImplicitGrid() {
  // checkCudaErrors(cudaMemset(d_grid_implicit, 0, sizeof(struct
  // TEST_STRUCT<T>)*_width*_height*_depth*_memoryScale));
}
}  // namespace mn
