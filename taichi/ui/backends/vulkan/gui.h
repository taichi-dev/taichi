#pragma once

#include "taichi/ui/utils/utils.h"

#ifndef IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#define IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#endif

#include <imgui.h>
#ifdef ANDROID
#include <imgui_impl_android.h>
#else
#include <imgui_impl_glfw.h>
#endif
#include <imgui_impl_vulkan.h>
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/common/gui_base.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class TI_DLL_EXPORT Gui final : public GuiBase {
 public:
  Gui(AppContext *app_context, SwapChain *swap_chain, TaichiWindow *window);
  ~Gui();
  void cleanup();

  void init_render_resources(VkRenderPass render_pass);
  void cleanup_render_resources();

  virtual void begin(std::string name,
                     float x,
                     float y,
                     float width,
                     float height) override;
  virtual void end() override;
  virtual void text(std::string text) override;
  virtual bool checkbox(std::string name, bool old_value) override;
  virtual int slider_int(std::string name,
                         int old_value,
                         int minimum,
                         int maximum) override;
  virtual float slider_float(std::string name,
                             float old_value,
                             float minimum,
                             float maximum) override;
  // TODO: consider renaming this?
  virtual glm::vec3 color_edit_3(std::string name,
                                 glm::vec3 old_value) override;
  virtual bool button(std::string text) override;

  void draw(taichi::lang::CommandList *cmd_list);

  void prepare_for_next_frame();

  VkRenderPass render_pass() {
    return render_pass_;
  }

  bool is_empty();

 private:
  bool is_empty_;
  AppContext *app_context_;
  SwapChain *swap_chain_;

  VkRenderPass render_pass_{VK_NULL_HANDLE};

  VkDescriptorPool descriptor_pool_;

  void create_descriptor_pool();

  float abs_x(float x);

  float abs_y(float y);

  bool initialized();
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
