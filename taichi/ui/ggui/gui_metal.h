#pragma once

#include "taichi/ui/utils/utils.h"

#include "taichi/ui/common/gui_base.h"
#include "taichi/ui/ggui/app_context.h"

#ifdef TI_WITH_METAL

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include "taichi/rhi/metal/metal_device.h"

namespace taichi::ui {

namespace vulkan {

class TI_DLL_EXPORT GuiMetal final : public GuiBase {
 public:
  GuiMetal(AppContext *app_context, TaichiWindow *window);
  ~GuiMetal() override;

  void init_render_resources(void *rpd);
  void cleanup_render_resources();

  void begin(const std::string &name,
             float x,
             float y,
             float width,
             float height) override;
  void end() override;
  void text(const std::string &text) override;
  void text(const std::string &text, glm::vec3 color) override;
  bool checkbox(const std::string &name, bool old_value) override;
  int slider_int(const std::string &name,
                 int old_value,
                 int minimum,
                 int maximum) override;
  float slider_float(const std::string &name,
                     float old_value,
                     float minimum,
                     float maximum) override;
  // TODO: consider renaming this?
  glm::vec3 color_edit_3(const std::string &name, glm::vec3 old_value) override;
  bool button(const std::string &text) override;

  void prepare_for_next_frame() override;

  void draw(taichi::lang::CommandList *cmd_list);

  bool is_empty();

 private:
  bool is_empty_;
  AppContext *app_context_{nullptr};
  ImGuiContext *imgui_context_{nullptr};
  int widthBeforeDPIScale{0};
  int heightBeforeDPIScale{0};

  MTLRenderPassDescriptor *current_rpd_{nullptr};

  float abs_x(float x);

  float abs_y(float y);
};

}  // namespace vulkan

}  // namespace taichi::ui

#endif
