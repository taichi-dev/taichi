#pragma once

#include <memory>
#include <vector>
#include "taichi/taichi_core.h"
#include "taichi/ui/gui/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"

namespace gui_api {

struct GuiHelper {
  explicit GuiHelper(TiArch arch,
                     const char *shader_path,
                     int window_h,
                     int window_w,
                     bool is_packed_mode);

  int set_circle_info(TiArch arch,
                      TiDataType dtype,
                      const std::vector<int> &shape,
                      const taichi::lang::DeviceAllocation &devalloc);

  int set_image_info(TiArch arch,
                     TiDataType dtype,
                     const std::vector<int> &shape,
                     const taichi::lang::DeviceAllocation &devalloc);

  void render_image(int handle);
  void render_circle(int handle);

  ~GuiHelper();

 private:
  std::shared_ptr<taichi::ui::vulkan::Gui> gui_{nullptr};
  std::unique_ptr<taichi::ui::vulkan::Renderer> renderer_{nullptr};
  GLFWwindow *window_{nullptr};
  std::map<int, taichi::ui::SetImageInfo> img_info_;
  std::map<int, taichi::ui::CirclesInfo> circle_info_;
};

}  // namespace gui_api
