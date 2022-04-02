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
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/backends/vulkan/scene.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Mesh final : public Renderable {
 public:
  Mesh(AppContext *app_context, VertexAttributes vbo_attrs);

  void update_data(const MeshInfo &info, const Scene &scene);

 private:
  struct UniformBufferObject {
    Scene::SceneUniformBuffer scene;
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
    int two_sided;
  };

  void init_mesh(AppContext *app_context,
                 int vertices_count,
                 int indices_count,
                 VertexAttributes vbo_attrs);

  void update_ubo(const MeshInfo &info, const Scene &scene);

  virtual void create_bindings() override;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
