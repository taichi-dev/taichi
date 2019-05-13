#include "GridUpdateKernels.cuh"
#include <System/CudaDevice/CudaDeviceUtils.cuh>
#include <cstdio>
#include <MnBase/Math/Matrix/MatrixKernels.cuh>
namespace mn {
__global__ void postP2G(T **d_channels) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  T mass = *((T *)((uint64_t)d_channels[0] + idx * 4096) + cellid);
  if (mass != 0.f) {
    mass = 1.f / mass;
    for (int i = 0; i < 3; ++i)
      *((T *)((uint64_t)d_channels[1 + i] + idx * 4096) + cellid) *= mass;
#if TRANSFER_SCHEME == 0
    for (int i = 0; i < 3; ++i)
      *((T *)((uint64_t)d_channels[7 + i] + idx * 4096) + cellid) =
          *((T *)((uint64_t)d_channels[1 + i] + idx * 4096) + cellid);
#endif
  }
}
__global__ void resetGrid(T **d_channels) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  *((T *)((uint64_t)d_channels[0] + idx * 4096) + cellid) = 0;
  if (true) {
    for (int i = 0; i < 3; ++i)
      *((T *)((uint64_t)d_channels[1 + i] + idx * 4096) + cellid) = 0;
  }
}
__global__ void applyGravity(const T dt, T **d_channels) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  if (*((T *)((uint64_t)d_channels[0] + idx * 4096) + cellid) != 0)
    *((T *)((uint64_t)d_channels[2] + idx * 4096) + cellid) += -1.0f * dt;
}
__global__ void updateVelocity(const T dt, T **d_channels) {
#if TRANSFER_SCHEME != 2
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  T mass = *((T *)((uint64_t)d_channels[0] + idx * 4096) + cellid);
  if (mass != 0.f) {
    mass = dt / mass;
    for (int i = 0; i < Dim; i++) {
      *((T *)((uint64_t)d_channels[i + 1] + idx * 4096) + cellid) +=
          *((T *)((uint64_t)d_channels[i + 4] + idx * 4096) + cellid) * mass;
    }
  }
#endif
}
// compute velocity diff : v0 = v - v0
// only apply to FLIP
__global__ void preG2P(T **d_channels) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  for (int i = 0; i < Dim; i++) {
    *((T *)((uint64_t)d_channels[i + 7] + idx * 4096) + cellid) =
        *((T *)((uint64_t)d_channels[i + 1] + idx * 4096) + cellid) -
        *((T *)((uint64_t)d_channels[i + 7] + idx * 4096) + cellid);
  }
}
__global__ void setFlags(
    ulonglong3 masks,
    uint64_t *pageOffsets,
    unsigned *d_implicit_flags)  // here should be implicit grid
{
  // one block corresponds to one page
  int cell = threadIdx.x;
  int ci = cell / 16;
  int ck = cell - ci * 16;
  int cj = ck / 4;
  ck = ck % 4;
  int i = Bit_Pack_Mine(masks.x, pageOffsets[blockIdx.x]) + ci;
  int j = Bit_Pack_Mine(masks.y, pageOffsets[blockIdx.x]) + cj;
  int k = Bit_Pack_Mine(masks.z, pageOffsets[blockIdx.x]) + ck;
  // x ground
  if (i <= 4 || i > N - 4 || j <= 4 || j > N - 4 || k <= 4 || k > N - 4)
    *((unsigned *)((uint64_t)d_implicit_flags + blockIdx.x * 4096) + cell) =
        MPM_DIRICHLET;
}
__global__ void implicitCopy(const int input_channel,
                             const T **input,
                             const int output_channel,
                             T **output) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  for (int v = 0; v < 3; ++v)
    *((T *)((uint64_t)output[output_channel + v] + idx * 4096) + cellid) =
        *((T *)((uint64_t)input[input_channel + v] + idx * 4096) + cellid);
}
__global__ void implicitScale(const T scale,
                              const T **input1,
                              const T **input2,
                              T **output) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  for (int v = 0; v < 3; ++v) {
    *((T *)((uint64_t)output[v] + idx * 4096) + cellid) =
        scale * (*((T *)((uint64_t)input1[v] + idx * 4096) + cellid)) +
        *((T *)((uint64_t)input2[v] + idx * 4096) + cellid);
  }
}
__global__ void implicitMinus(const T **input, T **inoutput) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  for (int v = 0; v < 3; ++v)
    *((T *)((uint64_t)inoutput[v] + idx * 4096) + cellid) -=
        *((T *)((uint64_t)input[v] + idx * 4096) + cellid);
}
__global__ void implicitProject(const unsigned *flags, T **output) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  if (*((unsigned *)((uint64_t)flags + idx * 4096) + cellid) & MPM_DIRICHLET) {
    for (int v = 0; v < 3; ++v)
      *((T *)((uint64_t)output[v] + idx * 4096) + cellid) = 0.f;
  }
}
__global__ void implicitClear(T **output) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  for (int v = 0; v < 3; ++v)
    *((T *)((uint64_t)output[v] + idx * 4096) + cellid) = 0.f;
}
__global__ void implicitSystemMatrix(const T scaled_dt_squared,
                                     T **d_channels,
                                     T **x,
                                     T **f) {
  int idx = blockIdx.x;
  int cellid = threadIdx.x;
  T mass = *((T *)((uint64_t)d_channels[0] + idx * 4096) + cellid);
  if (mass != 0) {
    mass = scaled_dt_squared / mass;
    for (int v = 0; v < 3; ++v)
      *((T *)((uint64_t)f[v] + idx * 4096) + cellid) =
          *((T *)((uint64_t)f[v] + idx * 4096) + cellid) * mass +
          *((T *)((uint64_t)x[v] + idx * 4096) + cellid);
  }
}
__global__ void implicitConvergenceNorm(const T **d_implicit_x,
                                        T *_maxNormSquared) {
  int idx = blockIdx.x;
  T local_max_squared = 0;
  int cellid = threadIdx.x;
  T data = (*((T *)((uint64_t)d_implicit_x[0] + idx * 4096) + cellid)) *
           (*((T *)((uint64_t)d_implicit_x[0] + idx * 4096) + cellid));
  data += (*((T *)((uint64_t)d_implicit_x[1] + idx * 4096) + cellid)) *
          (*((T *)((uint64_t)d_implicit_x[1] + idx * 4096) + cellid));
  data += (*((T *)((uint64_t)d_implicit_x[2] + idx * 4096) + cellid)) *
          (*((T *)((uint64_t)d_implicit_x[2] + idx * 4096) + cellid));
  if (data > local_max_squared)
    local_max_squared = data;
  for (uint32_t offset = 1; offset & 0x1f; offset <<= 1) {
    T tmp = __shfl_down(local_max_squared, offset);
    if (tmp > local_max_squared)
      local_max_squared = tmp;
  }
  if ((cellid & 0x1f) == 0)
    atomicMaxf(_maxNormSquared, local_max_squared);
}
#define warpSize 32
__global__ void implicitInnerProduct(const T **d_channels,
                                     const T **d_implicit_x,
                                     const T **d_implicit_y,
                                     double *_innerProduct,
                                     int N) {
  double sum = double(0);
  double mass = double(0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int idx = i / 64;
    int cellid = i % 64;
    mass = (*((T *)((uint64_t)d_channels[0] + idx * 4096) + cellid));
    sum += mass *
           (double)(*((T *)((uint64_t)d_implicit_x[0] + idx * 4096) + cellid)) *
           (double)(*((T *)((uint64_t)d_implicit_y[0] + idx * 4096) + cellid));
    sum += mass *
           (double)(*((T *)((uint64_t)d_implicit_x[1] + idx * 4096) + cellid)) *
           (double)(*((T *)((uint64_t)d_implicit_y[1] + idx * 4096) + cellid));
    sum += mass *
           (double)(*((T *)((uint64_t)d_implicit_x[2] + idx * 4096) + cellid)) *
           (double)(*((T *)((uint64_t)d_implicit_y[2] + idx * 4096) + cellid));
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    sum += __shfl_down(sum, offset);
  if (threadIdx.x % warpSize == 0)
    atomicAdd(_innerProduct, sum);
}
}  // namespace mn
