#include "G2PKernels.cuh"
#include <MnBase/Math/Matrix/MatrixKernels.cuh>
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <Simulation/ConstitutiveModel/ConstitutiveModelKernels.cuh>
#include <cstdio>
namespace mn {
__global__ void G2P_MLS(const int numParticle,
                        const int *d_targetPages,
                        const int *d_virtualPageOffsets,
                        const int **smallest_nodes,
                        int *d_block_offsets,
                        int *d_cellids,
                        int *d_indices,
                        int *d_indexTrans,
                        T **d_sorted_positions,
                        T **d_sorted_velocities,
                        T **d_channels,
                        T *d_sorted_F,
                        T *d_B,
                        T *d_tmp,
                        T dt,
                        int **d_adjPage) {
  __shared__ T buffer[3][8][8][8];
  int pageid = d_targetPages[blockIdx.x] - 1;  // from virtual to physical page
  int cellid = d_block_offsets[pageid];        //
  int relParid =
      512 * (blockIdx.x - d_virtualPageOffsets[pageid]) + threadIdx.x;
  int parid = cellid + relParid;
  int block = threadIdx.x & 0x3f;
  int ci = block >> 4;
  int cj = (block & 0xc) >> 2;
  int ck = block & 3;
  block = threadIdx.x >> 6;
  int bi = block >> 2;
  int bj = (block & 2) >> 1;
  int bk = block & 1;
  int page_idx = block ? d_adjPage[block - 1][pageid] : pageid;
  // vel
  for (int v = 0; v < 3; ++v)
    buffer[v][bi * 4 + ci][bj * 4 + cj][bk * 4 + ck] =
        *((T *)((uint64_t)d_channels[1 + v] + (int)page_idx * 4096) +
          (ci * 16 + cj * 4 + ck));
  __syncthreads();
  int smallest_node[3];
  if (relParid < d_block_offsets[pageid + 1] - d_block_offsets[pageid]) {
    cellid = d_cellids[parid] - 1;
    T wOneD[3][3];
    smallest_node[0] = smallest_nodes[0][cellid];
    smallest_node[1] = smallest_nodes[1][cellid];
    smallest_node[2] = smallest_nodes[2][cellid];
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
    int c = 0;
    float tmp[27];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          tmp[c++] = wOneD[0][i] * wOneD[1][j] * wOneD[2][k];
        }
      }
    }
    for (int v = 0; v < 3; ++v)
      smallest_node[v] = smallest_node[v] & 0x3;
    T val[9];
    for (int i = 0; i < 3; ++i)
      val[i] = 0.f;
    c = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          // v_pic
          val[0] += tmp[c] * buffer[0][smallest_node[0] + i]
                                   [smallest_node[1] + j][smallest_node[2] + k];
          val[1] += tmp[c] * buffer[1][smallest_node[0] + i]
                                   [smallest_node[1] + j][smallest_node[2] + k];
          val[2] += tmp[c++] *
                    buffer[2][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k];
        }
      }
    }
    d_tmp[parid] = val[0];
    d_tmp[parid + numParticle] = val[1];
    d_tmp[parid + numParticle * 2] = val[2];
    d_sorted_positions[0][parid] += val[0] * dt;
    d_sorted_positions[1][parid] += val[1] * dt;
    d_sorted_positions[2][parid] += val[2] * dt;
    for (int i = 0; i < 9; ++i)
      val[i] = 0.f;
    c = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          // B
          val[0] += tmp[c] *
                    buffer[0][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (i * dx - xp[0]);
          val[1] += tmp[c] *
                    buffer[1][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (i * dx - xp[0]);
          val[2] += tmp[c] *
                    buffer[2][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (i * dx - xp[0]);
          val[3] += tmp[c] *
                    buffer[0][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (j * dx - xp[1]);
          val[4] += tmp[c] *
                    buffer[1][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (j * dx - xp[1]);
          val[5] += tmp[c] *
                    buffer[2][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (j * dx - xp[1]);
          val[6] += tmp[c] *
                    buffer[0][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (k * dx - xp[2]);
          val[7] += tmp[c] *
                    buffer[1][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (k * dx - xp[2]);
          val[8] += tmp[c++] *
                    buffer[2][smallest_node[0] + i][smallest_node[1] + j]
                          [smallest_node[2] + k] *
                    (k * dx - xp[2]);
        }
      }
    }
    for (int i = 0; i < 9; ++i)
      d_tmp[parid + (i + 3) * numParticle] = val[i];
    for (int i = 0; i < 9; ++i)
      val[i] = val[i] * dt * D_inverse;
    val[0] += 1.f;
    val[4] += 1.f;
    val[8] += 1.f;
    T F[9];
    int parid_trans = d_indexTrans[parid];
    for (int i = 0; i < 9; ++i)
      F[i] = d_sorted_F[parid_trans + i * numParticle];
    T result[9];
    matrixMatrixMultiplication(&(val[0]), F, result);
    for (int i = 0; i < 9; ++i)
      d_tmp[parid + (i + 12) * numParticle] = result[i];
  }
}
}  // namespace mn
