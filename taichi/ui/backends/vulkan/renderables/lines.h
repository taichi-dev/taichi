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

namespace taichi::ui {

namespace vulkan {

class Lines final : public Renderable {
 public:
  Lines(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const LinesInfo &info);

  void create_graphics_pipeline() final;

  void record_prepass_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

 private:
  struct UniformBufferObject {
    alignas(16) glm::vec3 color;
    float line_width;
    int per_vertex_color_offset;
    int vertex_stride;
    int start_vertex;
    int start_index;
    int num_vertices;
    int is_indexed;
  };

  void init_lines(AppContext *app_context,
                  int vertices_count,
                  int indices_count);

  void update_ubo(glm::vec3 color, bool use_per_vertex_color);

  void cleanup() override;

  void create_bindings() override;

  float curr_width_;

  std::unique_ptr<taichi::lang::Pipeline> quad_expand_pipeline_{nullptr};

  std::unique_ptr<taichi::lang::DeviceAllocationGuard> vbo_translated_{nullptr};
  std::unique_ptr<taichi::lang::DeviceAllocationGuard> ibo_translated_{nullptr};
};

}  // namespace vulkan

}  // namespace taichi::ui
