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
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/vertex.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/ui/utils/utils.h"

namespace taichi {
namespace ui {
namespace vulkan {

using taichi::lang::DeviceAllocation;
using taichi::lang::DeviceAllocationUnique;
using taichi::lang::DevicePtr;
using taichi::lang::Pipeline;
using taichi::lang::RasterResources;
using taichi::lang::ShaderResourceSet;

struct RenderableConfig {
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
  bool depth{false};
  bool vertex_input_rate_instance{false};
};

class Renderable {
 public:
  bool is_3d_renderable{false};
  void update_data(const RenderableInfo &info);
  virtual void update_scene_data(DevicePtr ssbo_ptr, DevicePtr ubo_ptr);

  virtual void record_this_frame_commands(
      taichi::lang::CommandList *command_list);

  virtual void record_prepass_this_frame_commands(
      taichi::lang::CommandList *command_list) {
  }

  virtual ~Renderable() = default;

  taichi::lang::Pipeline &pipeline();
  const taichi::lang::Pipeline &pipeline() const;

  static void create_buffer_with_staging(taichi::lang::Device &device,
                                         size_t size,
                                         taichi::lang::AllocUsage usage,
                                         DeviceAllocationUnique &buffer,
                                         DeviceAllocationUnique &staging);

  static void copy_helper(taichi::lang::Program *prog,
                          DevicePtr dst,
                          DevicePtr src,
                          DevicePtr staging,
                          size_t size);

 protected:
  RenderableConfig config_;
  AppContext *app_context_;

  int max_vertices_count{0};
  int max_indices_count{0};

  Pipeline *pipeline_{nullptr};  // Factory owns pipelines
  std::unique_ptr<ShaderResourceSet> resource_set_{nullptr};

  DeviceAllocationUnique vertex_buffer_{nullptr};
  DeviceAllocationUnique index_buffer_{nullptr};

  // these staging buffers are used to copy data into the actual buffers
  DeviceAllocationUnique staging_vertex_buffer_{nullptr};
  DeviceAllocationUnique staging_index_buffer_{nullptr};

  DeviceAllocationUnique uniform_buffer_renderable_{nullptr};

  bool indexed_{false};

 protected:
  void init(const RenderableConfig &config_, AppContext *app_context);
  void init_buffers();

  virtual void create_graphics_pipeline();
};

}  // namespace vulkan
}  // namespace ui
}  // namespace taichi
