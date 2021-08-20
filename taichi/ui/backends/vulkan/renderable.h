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
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/backends/vulkan/vulkan_device.h"


TI_UI_NAMESPACE_BEGIN

namespace vulkan {

enum class TopologyType : int { Triangles = 0, Lines = 1, Points = 2 };

struct RenderableConfig {
  int vertices_count;
  int indices_count;
  size_t ubo_size;
  size_t ssbo_size;
  std::string vertex_shader_path;
  std::string fragment_shader_path;
  TopologyType topology_type;
};

class Renderable {
 public:
  void update_data(const RenderableInfo &info);

  virtual void record_this_frame_commands(VkCommandBuffer command_buffer);

  virtual void recreate_swap_chain();

  void cleanup_swap_chain();

  virtual void cleanup();

  virtual ~Renderable() = default;

 protected:
  RenderableConfig config_;

  class Renderer *renderer_;
  AppContext *app_context_;

  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;

  // TODO: use the memory allocator from ti vulkan backend
  taichi::lang::DeviceAllocation vertex_buffer_;
  taichi::lang::DeviceAllocation index_buffer_;

  // these staging buffers are used to copy data into the actual buffers when
  // `ti.cfg.arch==ti.cpu`
  VkBuffer staging_vertex_buffer_;
  VkDeviceMemory staging_vertex_buffer_memory_;
  VkBuffer staging_index_buffer_;
  VkDeviceMemory staging_index_buffer_memory_;

  VkBuffer uniform_buffer_;
  VkDeviceMemory uniform_buffer_memory_;

  VkBuffer storage_buffer_;
  VkDeviceMemory storage_buffer_memory_;

  VkDescriptorSetLayout descriptor_set_layout_;
  VkDescriptorSet descriptor_set_;

  VkDescriptorPool descriptor_pool_;

  Vertex *vertex_buffer_device_ptr_;
  int *index_buffer_device_ptr_;

  bool indexed_{false};

 protected:
  void init(const RenderableConfig &config_, class Renderer *renderer_);

  void init_render_resources();

  virtual void create_descriptor_set_layout() = 0;

  void create_descriptor_pool();

  void create_graphics_pipeline();

  void create_vertex_buffer();

  void create_index_buffer();

  void create_uniform_buffers();

  void create_storage_buffers();

  void destroy_uniform_buffers();

  void destroy_storage_buffers();

  void resize_storage_buffers(int new_ssbo_size);

  virtual void create_descriptor_sets() = 0;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
