#include "P2GKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <cstdio>
namespace mn {
__global__ void P2G_MLS(const int numParticle,
                        const int *d_targetPages,
                        const int *d_virtualPageOffsets,
                        const int **smallest_nodes,
                        const T *d_sigma,
                        int *d_block_offsets,
                        int *d_cellids,
                        int *d_indices,
                        int *d_indexTrans,
                        T **d_sorted_positions,
                        T *d_sorted_masses,
                        T **d_sorted_velocities,
                        T *d_B,
                        const T dt,
                        T **d_channels,
                        int **d_adjPage,
                        uint64_t *d_pageOffsets) {
  __shared__ T buffer[4][8][8][8];
  int cellid = (4 * 8 * 8 * 8 + blockDim.x - 1) / blockDim.x;
  for (int i = 0; i < cellid; ++i)
    if (blockDim.x * i + threadIdx.x < 4 * 8 * 8 * 8)
      *((&buffer[0][0][0][0]) + blockDim.x * i + threadIdx.x) = (T)0;
  __syncthreads();
  int pageid = d_targetPages[blockIdx.x] - 1;
  cellid = d_block_offsets[pageid];
  int relParid =
      512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
  int parid = cellid + relParid;
  int laneid = threadIdx.x & 0x1f;
  bool bBoundary;
  if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
    cellid = d_cellids[parid] - 1;
    bBoundary = laneid == 0 || cellid + 1 != d_cellids[parid - 1];
  } else
    bBoundary = true;
  uint32_t mark = __ballot(bBoundary);  // a bit-mask
  mark = __brev(mark);
  unsigned int interval = min(__clz(mark << (laneid + 1)), 31 - laneid);
  mark = interval;
  for (int iter = 1; iter & 0x1f; iter <<= 1) {
    int tmp = __shfl_down(mark, iter);
    mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
  }
  mark = __shfl(mark, 0);
  __syncthreads();
  int smallest_node[3];
  if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
    T wOneD[3][3];
    smallest_node[0] = smallest_nodes[0][cellid];
    smallest_node[1] = smallest_nodes[1][cellid];
    smallest_node[2] = smallest_nodes[2][cellid];
    int parid_trans = d_indexTrans[parid];
    T sig[9];
    sig[0] = d_sigma[parid_trans + (0) * numParticle];
    sig[1] = d_sigma[parid_trans + (1) * numParticle];
    sig[2] = d_sigma[parid_trans + (2) * numParticle];
    sig[3] = d_sigma[parid_trans + (3) * numParticle];
    sig[4] = d_sigma[parid_trans + (4) * numParticle];
    sig[5] = d_sigma[parid_trans + (5) * numParticle];
    sig[6] = d_sigma[parid_trans + (6) * numParticle];
    sig[7] = d_sigma[parid_trans + (7) * numParticle];
    sig[8] = d_sigma[parid_trans + (8) * numParticle];
    T B[9];
    B[0] = d_B[parid_trans + 0 * numParticle];
    B[1] = d_B[parid_trans + 1 * numParticle];
    B[2] = d_B[parid_trans + 2 * numParticle];
    B[3] = d_B[parid_trans + 3 * numParticle];
    B[4] = d_B[parid_trans + 4 * numParticle];
    B[5] = d_B[parid_trans + 5 * numParticle];
    B[6] = d_B[parid_trans + 6 * numParticle];
    B[7] = d_B[parid_trans + 7 * numParticle];
    B[8] = d_B[parid_trans + 8 * numParticle];
    T mass = d_sorted_masses[d_indices[parid]];
    for (int i = 0; i < 9; ++i)
      B[i] = (B[i] * mass - sig[i] * dt) * D_inverse;
    T xp[3];
    xp[0] = d_sorted_positions[0][parid] - smallest_node[0] * dx;
    xp[1] = d_sorted_positions[1][parid] - smallest_node[1] * dx;
    xp[2] = d_sorted_positions[2][parid] - smallest_node[2] * dx;
    for (int v = 0; v < 3; ++v) {
      T d0 = xp[v] * one_over_dx;
      T z = ((T)1.5 - d0);
      wOneD[v][0] = (T)0.5 * z * z;
      d0 = d0 - 1.0f;
      wOneD[v][1] = (T)0.75 - d0 * d0;
      z = (T)1.5 - (1.0f - d0);
      wOneD[v][2] = (T)0.5 * z * z;
    }
    T vel[3];
    vel[0] = d_sorted_velocities[0][parid_trans];
    vel[1] = d_sorted_velocities[1][parid_trans];
    vel[2] = d_sorted_velocities[2][parid_trans];
    smallest_node[0] = smallest_node[0] & 0x3;
    smallest_node[1] = smallest_node[1] & 0x3;
    smallest_node[2] = smallest_node[2] & 0x3;
    T val[4];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          T weight = wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
#ifdef DEBUG_INFO
          if (weight < 0 || weight > 1)
            printf("weight is negative!!! %f\n", weight);
#endif
          val[0] = mass * weight;
          T xi_minus_xp[3];
          xi_minus_xp[0] = i * dx - xp[0];
          xi_minus_xp[1] = j * dx - xp[1];
          xi_minus_xp[2] = k * dx - xp[2];
          val[1] = val[0] * vel[0];
          val[2] = val[0] * vel[1];
          val[3] = val[0] * vel[2];
          val[1] += (B[0] * xi_minus_xp[0] + B[3] * xi_minus_xp[1] +
                     B[6] * xi_minus_xp[2]) *
                    weight;
          val[2] += (B[1] * xi_minus_xp[0] + B[4] * xi_minus_xp[1] +
                     B[7] * xi_minus_xp[2]) *
                    weight;
          val[3] += (B[2] * xi_minus_xp[0] + B[5] * xi_minus_xp[1] +
                     B[8] * xi_minus_xp[2]) *
                    weight;
          for (int iter = 1; iter <= mark; iter <<= 1) {
            T tmp[4];
            for (int i = 0; i < 4; ++i)
              tmp[i] = __shfl_down(val[i], iter);
            if (interval >= iter)
              for (int i = 0; i < 4; ++i)
                val[i] += tmp[i];
          }
          if (bBoundary)
            for (int ii = 0; ii < 4; ++ii)
              atomicAdd(&(buffer[ii][smallest_node[0] + i][smallest_node[1] + j]
                                [smallest_node[2] + k]),
                        val[ii]);
        }
      }
    }
  }
  __syncthreads();
  int block = threadIdx.x & 0x3f;
  int ci = block >> 4;
  int cj = (block & 0xc) >> 2;
  int ck = block & 3;
  block = threadIdx.x >> 6;
  int bi = block >> 2;
  int bj = (block & 2) >> 1;
  int bk = block & 1;
  int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;
  for (int ii = 0; ii < 4; ++ii)
    if (buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] != 0)
      atomicAdd((T *)((uint64_t)d_channels[ii] + page_idx * 4096) +
                    (ci * 16 + cj * 4 + ck),
                buffer[ii][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck]);
}
}  // namespace mn
