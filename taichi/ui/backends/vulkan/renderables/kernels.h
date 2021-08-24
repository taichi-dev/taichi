#include "taichi/ui/backends/vulkan/vertex.h"
namespace taichi {
namespace ui {

void update_renderables_vertices_cuda(Vertex *vbo,
                                      float *vertices,
                                      int num_vertices,
                                      int num_components);
void update_renderables_vertices_x64(Vertex *vbo,
                                     float *vertices,
                                     int num_vertices,
                                     int num_components);

void update_renderables_indices_cuda(int *ibo, int *indices, int num_indices);
void update_renderables_indices_x64(int *ibo, int *indices, int num_indices);

void update_renderables_colors_cuda(Vertex *vbo,
                                    float *colors,
                                    int num_vertices);
void update_renderables_colors_x64(Vertex *vbo,
                                   float *colors,
                                   int num_vertices);

void update_renderables_normals_cuda(Vertex *vbo,
                                     float *normals,
                                     int num_vertices);
void update_renderables_normals_x64(Vertex *vbo,
                                    float *normals,
                                    int num_vertices);

template <typename T>
void copy_to_texture_buffer_surface_cuda(T *src,
                                         uint64_t surface,
                                         int width,
                                         int height,
                                         int actual_width,
                                         int actual_height,
                                         int channels);
template <typename T>
void copy_to_texture_buffer_cuda(T *src,
                                 unsigned char *dest,
                                 int width,
                                 int height,
                                 int actual_width,
                                 int actual_height,
                                 int channels);
template <typename T>
void copy_to_texture_buffer_x64(T *src,
                                unsigned char *dest,
                                int width,
                                int height,
                                int actual_width,
                                int actual_height,
                                int channels);

}  // namespace ui
}  // namespace taichi
