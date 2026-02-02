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
  glm::ivec2 slider_int2(const std::string &name,
                         glm::ivec2 old_value,
                         int minimum,
                         int maximum) override;
  glm::ivec3 slider_int3(const std::string &name,
                         glm::ivec3 old_value,
                         int minimum,
                         int maximum) override;
  glm::ivec4 slider_int4(const std::string &name,
                         glm::ivec4 old_value,
                         int minimum,
                         int maximum) override;
  float slider_float(const std::string &name,
                     float old_value,
                     float minimum,
                     float maximum) override;
  glm::vec2 slider_float2(const std::string &name,
                          glm::vec2 old_value,
                          float minimum,
                          float maximum) override;
  glm::vec3 slider_float3(const std::string &name,
                          glm::vec3 old_value,
                          float minimum,
                          float maximum) override;
  glm::vec4 slider_float4(const std::string &name,
                          glm::vec4 old_value,
                          float minimum,
                          float maximum) override;
  glm::vec3 color_edit_3(const std::string &name, glm::vec3 old_value) override;
  glm::vec4 color_edit_4(const std::string &name, glm::vec4 old_value) override;
  glm::vec3 color_picker_3(const std::string &name,
                           glm::vec3 old_value) override;
  glm::vec4 color_picker_4(const std::string &name,
                           glm::vec4 old_value) override;
  bool button(const std::string &text) override;
  int combo(const std::string &label,
            int current_item,
            const std::vector<const char *> &items) override;
  int input_int(const std::string &label, int old_value) override;
  glm::ivec2 input_int2(const std::string &label,
                        glm::ivec2 old_value) override;
  glm::ivec3 input_int3(const std::string &label,
                        glm::ivec3 old_value) override;
  glm::ivec4 input_int4(const std::string &label,
                        glm::ivec4 old_value) override;
  float input_float(const std::string &label, float old_value) override;
  glm::vec2 input_float2(const std::string &label,
                         glm::vec2 old_value) override;
  glm::vec3 input_float3(const std::string &label,
                         glm::vec3 old_value) override;
  glm::vec4 input_float4(const std::string &label,
                         glm::vec4 old_value) override;
  int drag_int(const std::string &label,
               int old_value,
               float speed,
               int minimum,
               int maximum) override;
  glm::ivec2 drag_int2(const std::string &label,
                       glm::ivec2 old_value,
                       float speed,
                       int minimum,
                       int maximum) override;
  glm::ivec3 drag_int3(const std::string &label,
                       glm::ivec3 old_value,
                       float speed,
                       int minimum,
                       int maximum) override;
  glm::ivec4 drag_int4(const std::string &label,
                       glm::ivec4 old_value,
                       float speed,
                       int minimum,
                       int maximum) override;
  float drag_float(const std::string &label,
                   float old_value,
                   float speed,
                   float minimum,
                   float maximum) override;
  glm::vec2 drag_float2(const std::string &label,
                        glm::vec2 old_value,
                        float speed,
                        float minimum,
                        float maximum) override;
  glm::vec3 drag_float3(const std::string &label,
                        glm::vec3 old_value,
                        float speed,
                        float minimum,
                        float maximum) override;
  glm::vec4 drag_float4(const std::string &label,
                        glm::vec4 old_value,
                        float speed,
                        float minimum,
                        float maximum) override;
  bool tree_node_push(const std::string &label) override;
  void tree_node_pop() override;
  void separator() override;
  void same_line() override;
  void indent() override;
  void unindent() override;
  void progress_bar(float fraction) override;
  bool collapsing_header(const std::string &label) override;
  bool selectable(const std::string &label, bool selected) override;
  bool radio_button(const std::string &label, bool active) override;
  int listbox(const std::string &label,
              int current_item,
              const std::vector<const char *> &items,
              int height_in_items) override;
  bool begin_tab_bar(const std::string &id) override;
  void end_tab_bar() override;
  bool begin_tab_item(const std::string &label) override;
  void end_tab_item() override;
  bool begin_table(const std::string &id, int columns) override;
  void end_table() override;
  void table_setup_column(const std::string &label) override;
  void table_headers_row() override;
  void table_next_row() override;
  bool table_next_column() override;

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
