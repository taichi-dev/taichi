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
#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/common/canvas_base.h"
#include "taichi/backends/device.h"


TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class SetImage final : public Renderable {
 public:
  int width, height;

  SetImage(class Renderer *renderer);

  void update_data(const SetImageInfo &info);

  virtual void cleanup() override;

 private:
  // the staging buffer is only used if we have a CPU ti backend.
  taichi::lang::DeviceAllocation  staging_buffer_;

  taichi::lang::DeviceAllocation texture_;

  uint64_t texture_surface_;

 private:
  void init_set_image(class Renderer *renderer, int img_width, int img_height);

  virtual void create_bindings() override;

  void create_texture();

  void create_texture_sampler();
  
  void update_vertex_buffer_();

  void update_index_buffer_();
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
