#include <device_launch_parameters.h>
#include "taichi/ui/backends/vulkan/vertex.h"

namespace taichi {
namespace ui {

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

__global__ void update_renderables_vertices_cuda_impl(Vertex *vbo,
                                                      float *vertices,
                                                      int num_vertices,
                                                      int num_components) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_vertices)
    return;

  vbo[i].pos.x = vertices[i * num_components];
  vbo[i].pos.y = vertices[i * num_components + 1];
  if (num_components == 3) {
    vbo[i].pos.z = vertices[i * num_components + 2];
  }
}

void update_renderables_vertices_cuda(Vertex *vbo,
                                      float *vertices,
                                      int num_vertices,
                                      int num_components) {
  int num_blocks, num_threads;
  set_num_blocks_threads(num_vertices, num_blocks, num_threads);
  update_renderables_vertices_cuda_impl<<<num_blocks, num_threads>>>(
      vbo, vertices, num_vertices, num_components);
}

void update_renderables_vertices_x64(Vertex *vbo,
                                     float *vertices,
                                     int num_vertices,
                                     int num_components) {
  for (int i = 0; i < num_vertices; ++i) {
    vbo[i].pos.x = vertices[i * num_components];
    vbo[i].pos.y = vertices[i * num_components + 1];
    if (num_components == 3) {
      vbo[i].pos.z = vertices[i * num_components + 2];
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

__global__ void update_renderables_colors_cuda_impl(Vertex *vbo,
                                                    float *colors,
                                                    int num_vertices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_vertices)
    return;

  vbo[i].color.x = colors[i * 3];
  vbo[i].color.y = colors[i * 3 + 1];
  vbo[i].color.z = colors[i * 3 + 2];
}
void update_renderables_colors_cuda(Vertex *vbo,
                                    float *colors,
                                    int num_vertices) {
  int num_blocks, num_threads;
  set_num_blocks_threads(num_vertices, num_blocks, num_threads);
  update_renderables_colors_cuda_impl<<<num_blocks, num_threads>>>(
      vbo, colors, num_vertices);
}

void update_renderables_colors_x64(Vertex *vbo,
                                   float *colors,
                                   int num_vertices) {
  for (int i = 0; i < num_vertices; ++i) {
    vbo[i].color.x = colors[i * 3];
    vbo[i].color.y = colors[i * 3 + 1];
    vbo[i].color.z = colors[i * 3 + 2];
  }
}

__global__ void update_renderables_normals_cuda_impl(Vertex *vbo,
                                                     float *normals,
                                                     int num_vertices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_vertices)
    return;

  vbo[i].normal.x = normals[i * 3];
  vbo[i].normal.y = normals[i * 3 + 1];
  vbo[i].normal.z = normals[i * 3 + 2];
}
void update_renderables_normals_cuda(Vertex *vbo,
                                     float *normals,
                                     int num_vertices) {
  int num_blocks, num_threads;
  set_num_blocks_threads(num_vertices, num_blocks, num_threads);
  update_renderables_normals_cuda_impl<<<num_blocks, num_threads>>>(
      vbo, normals, num_vertices);
}
void update_renderables_normals_x64(Vertex *vbo,
                                    float *normals,
                                    int num_vertices) {
  for (int i = 0; i < num_vertices; ++i) {
    vbo[i].normal.x = normals[i * 3];
    vbo[i].normal.y = normals[i * 3 + 1];
    vbo[i].normal.z = normals[i * 3 + 2];
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
__global__ void copy_to_texture_fuffer_cuda_impl(T *src,
                                                 uint64_t surface,
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

  surf3Dwrite(data, surface, x * sizeof(uchar4), y, 0);
}

template <typename T>
void copy_to_texture_fuffer_cuda(T *src,
                                 uint64_t surface,
                                 int width,
                                 int height,
                                 int actual_width,
                                 int actual_height,
                                 int channels) {
  int num_blocks, num_threads;
  set_num_blocks_threads(width * height, num_blocks, num_threads);
  copy_to_texture_fuffer_cuda_impl<<<num_blocks, num_threads>>>(
      src, (uint64_t)surface, width, height, actual_width, actual_height,
      channels);
}

template <typename T>
void copy_to_texture_fuffer_x64(T *src,
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

template void copy_to_texture_fuffer_cuda<float>(float *src,
                                                 uint64_t surface,
                                                 int width,
                                                 int height,
                                                 int actual_width,
                                                 int actual_height,
                                                 int channels);
template void copy_to_texture_fuffer_cuda<unsigned char>(unsigned char *src,
                                                         uint64_t surface,
                                                         int width,
                                                         int height,
                                                         int actual_width,
                                                         int actual_height,
                                                         int channels);

template void copy_to_texture_fuffer_x64<float>(float *src,
                                                unsigned char *dest,
                                                int width,
                                                int height,
                                                int actual_width,
                                                int actual_height,
                                                int channels);
template void copy_to_texture_fuffer_x64<unsigned char>(unsigned char *src,
                                                        unsigned char *dest,
                                                        int width,
                                                        int height,
                                                        int actual_width,
                                                        int actual_height,
                                                        int channels);

}  // namespace ui
}  // namespace taichi
