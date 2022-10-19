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
#include "taichi/ui/backends/vulkan/scene.h"

namespace taichi::ui {

namespace vulkan {

class SceneLines final : public Renderable {
 public:
  SceneLines(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const SceneLinesInfo &info, const Scene &scene);

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

 private:
  struct UniformBufferObject {
    Scene::SceneUniformBuffer scene;
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
  };

  void init_scene_lines(AppContext *app_context,
                        int vertices_count,
                        int indices_count);

  void update_ubo(const SceneLinesInfo &info, const Scene &scene);

  void cleanup() override;

  void create_bindings() override;

  float curr_width_;
};

}  // namespace vulkan

}  // namespace taichi::ui
