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
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/backends/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

struct RenderableConfig {
  int vertices_count;
  int indices_count;
  size_t ubo_size;
  size_t ssbo_size;
  std::string vertex_shader_path;
  std::string fragment_shader_path;
  taichi::lang::TopologyType topology_type;
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

  float *vertex_buffer_device_ptr_;
  int *index_buffer_device_ptr_;

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

TI_UI_NAMESPACE_END
