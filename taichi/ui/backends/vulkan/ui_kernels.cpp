
#include "taichi/inc/cuda_kernel_utils.inc.h"

extern "C" {

void update_renderables_vertices(float *vbo,
                                 int stride,
                                 float *data,
                                 int num_vertices,
                                 int num_components,
                                 int offset_bytes) {
  int i = block_idx() * block_dim() + thread_idx();

  float *dst = vbo + i * stride + offset_bytes / sizeof(float);
  float *src = data + i * num_components;
  for (int c = 0; c < num_components; ++c) {
    dst[c] = src[c];
  }
}

unsigned char get_color_value(float x) {
  x = x < 0 ? 0 : x;
  x = x > 1 ? 1 : x;
  return (unsigned char)(x * 255);
}

void copy_to_texture_buffer_u8(unsigned char *src,
                               unsigned char *dest,
                               int width,
                               int height,
                               int actual_width,
                               int actual_height,
                               int channels) {
  int i = block_idx() * block_dim() + thread_idx();
  if (i >= width * height)
    return;

  int y = i / width;
  int x = i % width;

  unsigned char *src_base_addr = src + (x * actual_height + y) * channels;
  unsigned char *dest_base_addr = dest + (y * width + x) * 4;
  dest_base_addr[0] = src_base_addr[0];
  dest_base_addr[1] = src_base_addr[1];
  dest_base_addr[2] = src_base_addr[2];
  dest_base_addr[3] = 255;
}

void copy_to_texture_buffer_f32(float *src,
                                unsigned char *dest,
                                int width,
                                int height,
                                int actual_width,
                                int actual_height,
                                int channels) {
  int i = block_idx() * block_dim() + thread_idx();
  if (i >= width * height)
    return;

  int y = i / width;
  int x = i % width;

  float *src_base_addr = src + (x * actual_height + y) * channels;
  unsigned char *dest_base_addr = dest + (y * width + x) * 4;
  dest_base_addr[0] = get_color_value(src_base_addr[0]);
  dest_base_addr[1] = get_color_value(src_base_addr[1]);
  dest_base_addr[2] = get_color_value(src_base_addr[2]);
  dest_base_addr[3] = 255;
}
}
