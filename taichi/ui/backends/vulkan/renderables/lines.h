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

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Lines final : public Renderable {
 public:
  Lines(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const LinesInfo &info);

  virtual void record_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

 private:
  struct UniformBufferObject {
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
  };

  void init_lines(AppContext *app_context,
                  int vertices_count,
                  int indices_count);

  void update_ubo(glm::vec3 color, bool use_per_vertex_color);

  virtual void cleanup() override;

  virtual void create_bindings() override;

  float curr_width_;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
