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

class Mesh final : public Renderable {
 public:
  Mesh(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const MeshInfo &info);
  void update_scene_data(DevicePtr ssbo_ptr, DevicePtr ubo_ptr) override;

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) override;

 private:
  DevicePtr lights_ssbo_ptr;
  DevicePtr scene_ubo_ptr;

  struct UBORenderable {
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
    int two_sided;
    float has_attribute;
  };

  int num_instances_{0};
  int start_instance_{0};

  size_t mesh_ssbo_size_{0};
  DeviceAllocationUnique mesh_storage_buffer_{nullptr};
  DeviceAllocationUnique mesh_staging_storage_buffer_{nullptr};

  void resize_mesh_storage_buffers(size_t ssbo_size);
};

}  // namespace vulkan

}  // namespace taichi::ui
