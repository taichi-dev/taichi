#include "MPMComputationKernels.cuh"
#include <cstdio>
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <MnBase/Math/Matrix/MatrixKernels.cuh>
#include <MnBase/Math/Matrix/svd3.h>
namespace mn {
__global__ void calcIndex(const int numCell,
                          const T one_over_dx,
                          const int *d_cell_first_particles_indices,
                          const T **d_sorted_positions,
                          int **smallest_nodes) {
  int cellid = blockDim.x * blockIdx.x + threadIdx.x;
  if (cellid >= numCell)
    return;
  smallest_nodes[0][cellid] =
      (int)((d_sorted_positions[0][d_cell_first_particles_indices[cellid]]) *
                one_over_dx +
            0.5f) -
      1;
  smallest_nodes[1][cellid] =
      (int)((d_sorted_positions[1][d_cell_first_particles_indices[cellid]]) *
                one_over_dx +
            0.5f) -
      1;
  smallest_nodes[2][cellid] =
      (int)((d_sorted_positions[2][d_cell_first_particles_indices[cellid]]) *
                one_over_dx +
            0.5f) -
      1;
}
__global__ void SVD3_Pre_Calculation(int numparticle,
                                     float *input,
                                     float *ouputdata) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= numparticle)
    return;
  int threadPerBlock = min(blockDim.x, numparticle);
  if (tid >= numparticle / blockDim.x * blockDim.x)
    threadPerBlock = numparticle % blockDim.x;
  __shared__ un sArray[504 * 21];
  // load to shared memory
  for (int i = 0; i < 9; i++) {
    int pos = i * threadPerBlock + threadIdx.x;
    int ipos = blockDim.x * 9 * blockIdx.x + pos;
    ipos = (ipos % 9) * numparticle + ipos / 9;
    sArray[pos / 9 * 21 + pos % 9].f = input[ipos];
  }
  __syncthreads();  // sync after load
  svd(sArray[threadIdx.x * 21 + 0].f, sArray[threadIdx.x * 21 + 3].f,
      sArray[threadIdx.x * 21 + 6].f, sArray[threadIdx.x * 21 + 1].f,
      sArray[threadIdx.x * 21 + 4].f, sArray[threadIdx.x * 21 + 7].f,
      sArray[threadIdx.x * 21 + 2].f, sArray[threadIdx.x * 21 + 5].f,
      sArray[threadIdx.x * 21 + 8].f, sArray[threadIdx.x * 21 + 0].f,
      sArray[threadIdx.x * 21 + 3].f, sArray[threadIdx.x * 21 + 6].f,
      sArray[threadIdx.x * 21 + 1].f, sArray[threadIdx.x * 21 + 4].f,
      sArray[threadIdx.x * 21 + 7].f, sArray[threadIdx.x * 21 + 2].f,
      sArray[threadIdx.x * 21 + 5].f, sArray[threadIdx.x * 21 + 8].f,
      sArray[threadIdx.x * 21 + 9].f, sArray[threadIdx.x * 21 + 10].f,
      sArray[threadIdx.x * 21 + 11].f, sArray[threadIdx.x * 21 + 12].f,
      sArray[threadIdx.x * 21 + 15].f, sArray[threadIdx.x * 21 + 18].f,
      sArray[threadIdx.x * 21 + 13].f, sArray[threadIdx.x * 21 + 16].f,
      sArray[threadIdx.x * 21 + 19].f, sArray[threadIdx.x * 21 + 14].f,
      sArray[threadIdx.x * 21 + 17].f, sArray[threadIdx.x * 21 + 20].f);
  __syncthreads();  // sync before store
  for (int i = 0; i < 21; i++) {
    ouputdata[blockDim.x * 21 * blockIdx.x + i * threadPerBlock + threadIdx.x] =
        sArray[i * threadPerBlock + threadIdx.x].f;
  }
}
__global__ void computeContributionFixedCorotated(const int numParticle,
                                                  const T *d_F,
                                                  const T lambda,
                                                  const T mu,
                                                  const T volume,
                                                  T *d_contribution) {
  int parid = blockDim.x * blockIdx.x + threadIdx.x;
  if (parid >= numParticle)
    return;
  T F[9];
  F[0] = d_F[parid + 0 * numParticle];
  F[1] = d_F[parid + 1 * numParticle];
  F[2] = d_F[parid + 2 * numParticle];
  F[3] = d_F[parid + 3 * numParticle];
  F[4] = d_F[parid + 4 * numParticle];
  F[5] = d_F[parid + 5 * numParticle];
  F[6] = d_F[parid + 6 * numParticle];
  F[7] = d_F[parid + 7 * numParticle];
  F[8] = d_F[parid + 8 * numParticle];
  T U[9];
  T S[3];
  T V[9];
  svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6],
      U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6],
      V[1], V[4], V[7], V[2], V[5], V[8]);
  //
  T J = S[0] * S[1] * S[2];
  T scaled_mu = 2.f * mu;
  T scaled_lambda = lambda * (J - 1.f);
  T P_hat[3];
  P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
  P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
  P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);
  //
  T P[9];
  P[0] =
      P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6];
  P[1] =
      P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6];
  P[2] =
      P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
  P[3] =
      P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7];
  P[4] =
      P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7];
  P[5] =
      P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
  P[6] =
      P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8];
  P[7] =
      P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8];
  P[8] =
      P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];
  d_contribution[parid + 0 * numParticle] =
      (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
  d_contribution[parid + 1 * numParticle] =
      (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
  d_contribution[parid + 2 * numParticle] =
      (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
  d_contribution[parid + 3 * numParticle] =
      (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
  d_contribution[parid + 4 * numParticle] =
      (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
  d_contribution[parid + 5 * numParticle] =
      (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
  d_contribution[parid + 6 * numParticle] =
      (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
  d_contribution[parid + 7 * numParticle] =
      (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
  d_contribution[parid + 8 * numParticle] =
      (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}
__global__ void collideWithGround(ulonglong3 masks,
                                  uint64_t *pageOffsets,
                                  T **d_channels) {
  // one block corresponds to one page
  int cell = threadIdx.x;
  int ci = cell / 16;
  int ck = cell - ci * 16;
  int cj = ck / 4;
  ck = ck % 4;
  int i = Bit_Pack_Mine(masks.x, pageOffsets[blockIdx.x]) + ci;
  int j = Bit_Pack_Mine(masks.y, pageOffsets[blockIdx.x]) + cj;
  int k = Bit_Pack_Mine(masks.z, pageOffsets[blockIdx.x]) + ck;
  // sticky
  if (i <= 4)
    if (*((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) < 0.f) {
      *((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) = 0.f;
    }
  if (i > N - 4)
    if (*((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) > 0.f) {
      *((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) = 0.f;
    }
  if (j <= 4)
    if (*((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) < 0.f) {
      *((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) = 0.f;
    }
  if (j > N - 4)
    if (*((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) > 0.f) {
      *((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) = 0.f;
    }
  if (k <= 4)
    if (*((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) < 0.f) {
      *((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) = 0.f;
    }
  if (k > N - 4)
    if (*((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) > 0.f) {
      *((T *)((uint64_t)d_channels[3] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[1] + blockIdx.x * 4096) + cell) = 0.f;
      *((T *)((uint64_t)d_channels[2] + blockIdx.x * 4096) + cell) = 0.f;
    }
}
}  // namespace mn
