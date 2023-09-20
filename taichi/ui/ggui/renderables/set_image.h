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
#include "taichi/ui/common/canvas_base.h"
#include "taichi/rhi/device.h"

namespace taichi::ui {

namespace vulkan {

class SetImage final : public Renderable {
 public:
  struct UniformBufferObject {
    glm::vec2 lower_bound;
    glm::vec2 upper_bound;
    // in non_packed_mode,
    // the actual image is only a corner of the whole image
    float x_factor{1.0};
    float y_factor{1.0};
    int transpose{0};
  };

  SetImage(AppContext *app_context, VertexAttributes vbo_attrs);

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) final;

  void update_data(const SetImageInfo &info);

  void update_data(taichi::lang::Texture *tex);

 private:
  int width_{0};
  int height_{0};

  taichi::lang::DeviceImageUnique texture_{nullptr};

  taichi::lang::BufferFormat format_;

 private:
  void resize_texture(int width, int height, taichi::lang::BufferFormat format);

  void update_ubo(float x_factor, float y_factor, bool transpose);
};

}  // namespace vulkan

}  // namespace taichi::ui
