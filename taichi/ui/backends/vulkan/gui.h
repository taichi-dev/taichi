#pragma once

#include "taichi/ui/utils/utils.h"

#ifndef IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#define IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#endif

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/common/gui_base.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Gui final : public GuiBase {
 public:
  void init(AppContext *app_context_, GLFWwindow *window);

  virtual void begin(std::string name,
                     float x,
                     float y,
                     float width,
                     float height) override;
  virtual void end() override;
  virtual void text(std::string text) override;
  virtual bool checkbox(std::string name, bool old_value) override;
  virtual float slider_float(std::string name,
                             float old_value,
                             float minimum,
                             float maximum) override;
  virtual glm::vec3 color_edit_3(std::string name,
                                 glm::vec3 old_value) override;
  virtual bool button(std::string text) override;

  void draw(VkCommandBuffer &command_buffer);
  void cleanup();

  void prepare_for_next_frame();

  bool is_empty;

 private:
  AppContext *app_context_;

  VkDescriptorPool descriptor_pool_;

  void create_descriptor_pool();

  float abs_x(float x);

  float abs_y(float y);
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
