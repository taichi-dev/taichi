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

class Particles final : public Renderable {
 public:
  Particles(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const ParticlesInfo &info, const Scene &scene);

  void record_this_frame_commands(lang::CommandList *command_list) override;

 private:
  struct UniformBufferObject {
    Scene::SceneUniformBuffer scene;
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
    float radius;
    float window_width;
    float window_height;
    float tan_half_fov;
  };
};

}  // namespace vulkan

}  // namespace taichi::ui
