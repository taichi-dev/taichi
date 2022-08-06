#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/vertex.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/ui/utils/utils.h"

namespace taichi {
namespace ui {
namespace vulkan {

struct RenderableConfig {
  int max_vertices_count{0};
  int max_indices_count{0};
  int vertices_count{0};
  int indices_count{0};
  int draw_vertex_count{0};
  int draw_first_vertex{0};
  int draw_index_count{0};
  int draw_first_index{0};
  size_t ubo_size{0};
  size_t ssbo_size{0};
  bool blending{false};
  std::string vertex_shader_path;
  std::string fragment_shader_path;
  taichi::lang::TopologyType topology_type{
      taichi::lang::TopologyType::Triangles};
  taichi::lang::PolygonMode polygon_mode{taichi::lang::PolygonMode::Fill};
  VertexAttributes vbo_attrs{VboHelpers::all()};

  size_t vbo_size() const {
    return VboHelpers::size(vbo_attrs);
  }
};

class Renderable {
 public:
  void update_data(const RenderableInfo &info);

  virtual void record_this_frame_commands(
      taichi::lang::CommandList *command_list);

  virtual ~Renderable() = default;

  taichi::lang::Pipeline &pipeline();
  const taichi::lang::Pipeline &pipeline() const;

  virtual void cleanup();

 protected:
  RenderableConfig config_;
  AppContext *app_context_;

  std::unique_ptr<taichi::lang::Pipeline> pipeline_{nullptr};

  taichi::lang::DeviceAllocation vertex_buffer_;
  taichi::lang::DeviceAllocation index_buffer_;

  // these staging buffers are used to copy data into the actual buffers when
  // `ti.cfg.arch==ti.cpu`
  taichi::lang::DeviceAllocation staging_vertex_buffer_;
  taichi::lang::DeviceAllocation staging_index_buffer_;

  taichi::lang::DeviceAllocation uniform_buffer_;
  taichi::lang::DeviceAllocation storage_buffer_;

  bool indexed_{false};

 protected:
  void init(const RenderableConfig &config_, AppContext *app_context);
  void free_buffers();
  void init_buffers();
  void init_render_resources();

  virtual void create_bindings();

  void create_graphics_pipeline();

  void create_vertex_buffer();

  void create_index_buffer();

  void create_uniform_buffers();

  void create_storage_buffers();

  void destroy_uniform_buffers();

  void destroy_storage_buffers();

  void resize_storage_buffers(int new_ssbo_size);
};

}  // namespace vulkan
}  // namespace ui
}  // namespace taichi
