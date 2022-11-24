#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vertex.h"

#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/common/canvas_base.h"
#include "taichi/rhi/device.h"

namespace taichi::ui {

namespace vulkan {

class SetImage final : public Renderable {
 public:
  int width, height;

  struct UniformBufferObject {
    // in non_packed_mode,
    // the actual image is only a corner of the whole image
    float x_factor{1.0};
    float y_factor{1.0};
    int transpose{0};
  };

  SetImage(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const SetImageInfo &info);

  void update_data(taichi::lang::Texture *tex);

  void cleanup() override;

 private:
  taichi::lang::DeviceAllocation cpu_staging_buffer_;
  taichi::lang::DeviceAllocation gpu_staging_buffer_;

  taichi::lang::DataType texture_dtype_{taichi::lang::PrimitiveType::u8};
  taichi::lang::DeviceAllocation texture_;

  taichi::lang::BufferFormat format_;

 private:
  void init_set_image(AppContext *app_context,
                      int img_width,
                      int img_height,
                      taichi::lang::BufferFormat format);

  void create_bindings() override;

  void create_texture();
  void destroy_texture();

  void update_vertex_buffer();

  void update_index_buffer();

  int get_correct_dimension(int dimension);

  void update_ubo(float x_factor, float y_factor, bool transpose);
};

}  // namespace vulkan

}  // namespace taichi::ui
