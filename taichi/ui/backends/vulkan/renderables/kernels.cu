#include <device_launch_parameters.h>
#include "taichi/ui/backends/vulkan/vertex.h"

namespace taichi {
namespace ui {

namespace {
int div_up(int a, int b) {
  if (b == 0) {
    return 1;
  }
  int result = (a % b != 0) ? (a / b + 1) : (a / b);
  return result;
}

#define MAX_THREADS_PER_BLOCK 1024
void set_num_blocks_threads(int N, int &num_blocks, int &num_threads) {
  num_threads = min(N, MAX_THREADS_PER_BLOCK);
  num_blocks = div_up(N, num_threads);
}
#undef MAX_THREADS_PER_BLOCK
}  // namespace

__global__ void update_renderables_vertices_cuda_impl(Vertex *vbo,
                                                      float *data,
                                                      int num_vertices,
                                                      int num_components,
                                                      int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_vertices)
    return;

  float *dst = (float *)(vbo + i) + offset;
  float *src = data + i * num_components;
  for (int c = 0; c < num_components; ++c) {
    dst[c] = src[c];
  }
}

void update_renderables_vertices_cuda(Vertex *vbo,
                                      float *data,
                                      int num_vertices,
                                      int num_components,
                                      int offset_bytes) {
  int num_blocks, num_threads;
  set_num_blocks_threads(num_vertices, num_blocks, num_threads);
  update_renderables_vertices_cuda_impl<<<num_blocks, num_threads>>>(
      vbo, data, num_vertices, num_components, offset_bytes / sizeof(float));
}

void update_renderables_vertices_x64(Vertex *vbo,
                                     float *data,
                                     int num_vertices,
                                     int num_components,
                                     int offset_bytes) {
  int offset = offset_bytes / sizeof(float);
  for (int i = 0; i < num_vertices; ++i) {
    float *dst = (float *)(vbo + i) + offset;
    float *src = data + i * num_components;
    for (int c = 0; c < num_components; ++c) {
      dst[c] = src[c];
    }
  }
}

__global__ void update_renderables_indices_cuda_impl(int *ibo,
                                                     int *indices,
                                                     int num_indices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_indices)
    return;

  ibo[i] = indices[i];
}
void update_renderables_indices_cuda(int *ibo, int *indices, int num_indices) {
  int num_blocks, num_threads;
  set_num_blocks_threads(num_indices, num_blocks, num_threads);
  update_renderables_indices_cuda_impl<<<num_blocks, num_threads>>>(
      ibo, indices, num_indices);
}

void update_renderables_indices_x64(int *ibo, int *indices, int num_indices) {
  for (int i = 0; i < num_indices; ++i) {
    ibo[i] = indices[i];
  }
}

template <typename T>
__device__ __host__ inline unsigned char get_color_value(T x);

template <>
__device__ __host__ inline unsigned char get_color_value<unsigned char>(
    unsigned char x) {
  return x;
}

template <>
__device__ __host__ inline unsigned char get_color_value<float>(float x) {
  x = max(0.f, min(1.f, x));
  return (unsigned char)(x * 255);
}

template <typename T>
void copy_to_texture_buffer_x64(T *src,
                                unsigned char *dest,
                                int width,
                                int height,
                                int actual_width,
                                int actual_height,
                                int channels) {
  for (int i = 0; i < width * height; ++i) {
    int y = i / width;
    int x = i % width;

    T *src_base_addr = src + (x * actual_height + y) * channels;
    uchar4 data = make_uchar4(0, 0, 0, 0);

    data.x = get_color_value<T>(src_base_addr[0]);
    data.y = get_color_value<T>(src_base_addr[1]);
    data.z = get_color_value<T>(src_base_addr[2]);
    data.w = 255;

    ((uchar4 *)dest)[y * width + x] = data;
  }
}

template <typename T>
__global__ void copy_to_texture_buffer_cuda_impl(T *src,
                                                 unsigned char *dest,
                                                 int width,
                                                 int height,
                                                 int actual_width,
                                                 int actual_height,
                                                 int channels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height)
    return;

  int y = i / width;
  int x = i % width;

  T *src_base_addr = src + (x * actual_height + y) * channels;
  uchar4 data = make_uchar4(0, 0, 0, 0);

  data.x = get_color_value<T>(src_base_addr[0]);
  data.y = get_color_value<T>(src_base_addr[1]);
  data.z = get_color_value<T>(src_base_addr[2]);
  data.w = 255;

  ((uchar4 *)dest)[y * width + x] = data;
}

template <typename T>
void copy_to_texture_buffer_cuda(T *src,
                                 unsigned char *dest,
                                 int width,
                                 int height,
                                 int actual_width,
                                 int actual_height,
                                 int channels) {
  int num_blocks, num_threads;
  set_num_blocks_threads(width * height, num_blocks, num_threads);
  copy_to_texture_buffer_cuda_impl<<<num_blocks, num_threads>>>(
      src, dest, width, height, actual_width, actual_height, channels);
}

template void copy_to_texture_buffer_cuda<float>(float *src,
                                                 unsigned char *dest,
                                                 int width,
                                                 int height,
                                                 int actual_width,
                                                 int actual_height,
                                                 int channels);
template void copy_to_texture_buffer_cuda<unsigned char>(unsigned char *src,
                                                         unsigned char *dest,
                                                         int width,
                                                         int height,
                                                         int actual_width,
                                                         int actual_height,
                                                         int channels);
template void copy_to_texture_buffer_x64<float>(float *src,
                                                unsigned char *dest,
                                                int width,
                                                int height,
                                                int actual_width,
                                                int actual_height,
                                                int channels);
template void copy_to_texture_buffer_x64<unsigned char>(unsigned char *src,
                                                        unsigned char *dest,
                                                        int width,
                                                        int height,
                                                        int actual_width,
                                                        int actual_height,
                                                        int channels);

}  // namespace ui
}  // namespace taichi
