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
#include "taichi/ui/ggui/vertex.h"

#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/renderable.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/ggui/scene.h"

namespace taichi::ui {

namespace vulkan {

class SceneLines final : public Renderable {
 public:
  SceneLines(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const SceneLinesInfo &info);

  void update_scene_data(DevicePtr ssbo_ptr, DevicePtr ubo_ptr) override;

  void record_prepass_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

 private:
  DevicePtr lights_ssbo_ptr;
  DevicePtr scene_ubo_ptr;

  struct UBORenderable {
    alignas(16) glm::vec3 color;
    float line_width;
    int per_vertex_color_offset;
    int vertex_stride;
    int start_vertex;
    int start_index;
    int num_vertices;
    int is_indexed;
  };

  void create_graphics_pipeline() final;

  uint64_t lines_count_{0};

  taichi::lang::Pipeline *quad_expand_pipeline_{nullptr};

  std::unique_ptr<taichi::lang::DeviceAllocationGuard> vbo_translated_{nullptr};
  std::unique_ptr<taichi::lang::DeviceAllocationGuard> ibo_translated_{nullptr};
};

}  // namespace vulkan

}  // namespace taichi::ui
