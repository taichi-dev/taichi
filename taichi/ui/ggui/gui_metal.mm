#include "gui_metal.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include <imgui_impl_metal.h>

using namespace taichi::lang::metal;
using namespace taichi::lang;

namespace taichi::ui {

namespace vulkan {

GuiMetal::GuiMetal(AppContext *app_context, TaichiWindow *window) {
  app_context_ = app_context;

  IMGUI_CHECKVERSION();
  imgui_context_ = ImGui::CreateContext();
  [[maybe_unused]] ImGuiIO &io = ImGui::GetIO();

  ImGui::StyleColorsDark();

  if (app_context->config.show_window) {
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    glfwGetWindowSize(window, &widthBeforeDPIScale, &heightBeforeDPIScale);
  } else {
    widthBeforeDPIScale = app_context->config.width;
    heightBeforeDPIScale = app_context->config.height;
  }
  auto &device =
      static_cast<taichi::lang::metal::MetalDevice &>(app_context_->device());

  ImGui_ImplMetal_Init(device.mtl_device());
}

void GuiMetal::init_render_resources(void *rpd) {
  current_rpd_ = (__bridge MTLRenderPassDescriptor *)rpd;
}

void GuiMetal::prepare_for_next_frame() {
  if (app_context_->config.show_window) {
    ImGui_ImplGlfw_NewFrame();
  } else {
    // io.DisplaySize is set during ImGui_ImplGlfw_NewFrame()
    // but since we're headless, we do it explicitly here
    auto w = app_context_->config.width;
    auto h = app_context_->config.height;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)w, (float)h);
  }
  ImGui::NewFrame();
  is_empty_ = true;
}

float GuiMetal::abs_x(float x) { return x * widthBeforeDPIScale; }
float GuiMetal::abs_y(float y) { return y * heightBeforeDPIScale; }

void GuiMetal::begin(const std::string &name, float x, float y, float width,
                     float height) {
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(abs_x(width), abs_y(height)), ImGuiCond_Once);
  ImGui::Begin(name.c_str());
  is_empty_ = false;
}
void GuiMetal::end() { ImGui::End(); }
void GuiMetal::text(const std::string &text) {
  ImGui::Text("%s", text.c_str());
}
void GuiMetal::text(const std::string &text, glm::vec3 color) {
  ImGui::TextColored(ImVec4(color[0], color[1], color[2], 1.0f), "%s",
                     text.c_str());
}
bool GuiMetal::checkbox(const std::string &name, bool old_value) {
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}
int GuiMetal::slider_int(const std::string &name, int old_value, int minimum,
                         int maximum) {
  ImGui::SliderInt(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::ivec2 GuiMetal::slider_int2(const std::string &name, glm::ivec2 old_value,
                                 int minimum, int maximum) {
  ImGui::SliderInt2(name.c_str(), (int *)&old_value, minimum, maximum);
  return old_value;
}
glm::ivec3 GuiMetal::slider_int3(const std::string &name, glm::ivec3 old_value,
                                 int minimum, int maximum) {
  ImGui::SliderInt3(name.c_str(), (int *)&old_value, minimum, maximum);
  return old_value;
}
glm::ivec4 GuiMetal::slider_int4(const std::string &name, glm::ivec4 old_value,
                                 int minimum, int maximum) {
  ImGui::SliderInt4(name.c_str(), (int *)&old_value, minimum, maximum);
  return old_value;
}
float GuiMetal::slider_float(const std::string &name, float old_value,
                             float minimum, float maximum) {
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::vec2 GuiMetal::slider_float2(const std::string &name, glm::vec2 old_value,
                                  float minimum, float maximum) {
  ImGui::SliderFloat2(name.c_str(), (float *)&old_value, minimum, maximum);
  return old_value;
}
glm::vec3 GuiMetal::slider_float3(const std::string &name, glm::vec3 old_value,
                                  float minimum, float maximum) {
  ImGui::SliderFloat3(name.c_str(), (float *)&old_value, minimum, maximum);
  return old_value;
}
glm::vec4 GuiMetal::slider_float4(const std::string &name, glm::vec4 old_value,
                                  float minimum, float maximum) {
  ImGui::SliderFloat4(name.c_str(), (float *)&old_value, minimum, maximum);
  return old_value;
}
glm::vec3 GuiMetal::color_edit_3(const std::string &name, glm::vec3 old_value) {
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}
glm::vec4 GuiMetal::color_edit_4(const std::string &name, glm::vec4 old_value) {
  ImGui::ColorEdit4(name.c_str(), (float *)&old_value);
  return old_value;
}
glm::vec3 GuiMetal::color_picker_3(const std::string &name,
                                   glm::vec3 old_value) {
  ImGui::ColorPicker3(name.c_str(), (float *)&old_value);
  return old_value;
}
glm::vec4 GuiMetal::color_picker_4(const std::string &name,
                                   glm::vec4 old_value) {
  ImGui::ColorPicker4(name.c_str(), (float *)&old_value);
  return old_value;
}
bool GuiMetal::button(const std::string &text) {
  return ImGui::Button(text.c_str());
}

int GuiMetal::combo(const std::string &label, int current_item,
                    const std::vector<const char *> &items) {
  ImGui::Combo(label.c_str(), &current_item, items.data(),
               static_cast<int>(items.size()));
  return current_item;
}

int GuiMetal::input_int(const std::string &label, int old_value) {
  ImGui::InputInt(label.c_str(), &old_value);
  return old_value;
}

glm::ivec2 GuiMetal::input_int2(const std::string &label,
                                glm::ivec2 old_value) {
  ImGui::InputInt2(label.c_str(), (int *)&old_value);
  return old_value;
}

glm::ivec3 GuiMetal::input_int3(const std::string &label,
                                glm::ivec3 old_value) {
  ImGui::InputInt3(label.c_str(), (int *)&old_value);
  return old_value;
}

glm::ivec4 GuiMetal::input_int4(const std::string &label,
                                glm::ivec4 old_value) {
  ImGui::InputInt4(label.c_str(), (int *)&old_value);
  return old_value;
}

float GuiMetal::input_float(const std::string &label, float old_value) {
  ImGui::InputFloat(label.c_str(), &old_value);
  return old_value;
}

glm::vec2 GuiMetal::input_float2(const std::string &label,
                                 glm::vec2 old_value) {
  ImGui::InputFloat2(label.c_str(), (float *)&old_value);
  return old_value;
}

glm::vec3 GuiMetal::input_float3(const std::string &label,
                                 glm::vec3 old_value) {
  ImGui::InputFloat3(label.c_str(), (float *)&old_value);
  return old_value;
}

glm::vec4 GuiMetal::input_float4(const std::string &label,
                                 glm::vec4 old_value) {
  ImGui::InputFloat4(label.c_str(), (float *)&old_value);
  return old_value;
}

int GuiMetal::drag_int(const std::string &label, int old_value, float speed,
                       int minimum, int maximum) {
  ImGui::DragInt(label.c_str(), &old_value, speed, minimum, maximum);
  return old_value;
}

glm::ivec2 GuiMetal::drag_int2(const std::string &label, glm::ivec2 old_value,
                               float speed, int minimum, int maximum) {
  ImGui::DragInt2(label.c_str(), (int *)&old_value, speed, minimum, maximum);
  return old_value;
}

glm::ivec3 GuiMetal::drag_int3(const std::string &label, glm::ivec3 old_value,
                               float speed, int minimum, int maximum) {
  ImGui::DragInt3(label.c_str(), (int *)&old_value, speed, minimum, maximum);
  return old_value;
}

glm::ivec4 GuiMetal::drag_int4(const std::string &label, glm::ivec4 old_value,
                               float speed, int minimum, int maximum) {
  ImGui::DragInt4(label.c_str(), (int *)&old_value, speed, minimum, maximum);
  return old_value;
}

float GuiMetal::drag_float(const std::string &label, float old_value,
                           float speed, float minimum, float maximum) {
  ImGui::DragFloat(label.c_str(), &old_value, speed, minimum, maximum);
  return old_value;
}

glm::vec2 GuiMetal::drag_float2(const std::string &label, glm::vec2 old_value,
                                float speed, float minimum, float maximum) {
  ImGui::DragFloat2(label.c_str(), (float *)&old_value, speed, minimum,
                    maximum);
  return old_value;
}

glm::vec3 GuiMetal::drag_float3(const std::string &label, glm::vec3 old_value,
                                float speed, float minimum, float maximum) {
  ImGui::DragFloat3(label.c_str(), (float *)&old_value, speed, minimum,
                    maximum);
  return old_value;
}

glm::vec4 GuiMetal::drag_float4(const std::string &label, glm::vec4 old_value,
                                float speed, float minimum, float maximum) {
  ImGui::DragFloat4(label.c_str(), (float *)&old_value, speed, minimum,
                    maximum);
  return old_value;
}

bool GuiMetal::tree_node_push(const std::string &label) {
  return ImGui::TreeNode(label.c_str());
}

void GuiMetal::tree_node_pop() { ImGui::TreePop(); }

void GuiMetal::separator() { ImGui::Separator(); }

void GuiMetal::same_line() { ImGui::SameLine(); }

void GuiMetal::indent() { ImGui::Indent(); }

void GuiMetal::unindent() { ImGui::Unindent(); }

void GuiMetal::progress_bar(float fraction) { ImGui::ProgressBar(fraction); }

bool GuiMetal::collapsing_header(const std::string &label) {
  return ImGui::CollapsingHeader(label.c_str());
}

bool GuiMetal::selectable(const std::string &label, bool selected) {
  ImGui::Selectable(label.c_str(), &selected);
  return selected;
}

bool GuiMetal::radio_button(const std::string &label, bool active) {
  return ImGui::RadioButton(label.c_str(), active);
}

int GuiMetal::listbox(const std::string &label, int current_item,
                      const std::vector<const char *> &items,
                      int height_in_items) {
  ImGui::ListBox(label.c_str(), &current_item, items.data(),
                 static_cast<int>(items.size()), height_in_items);
  return current_item;
}

bool GuiMetal::begin_tab_bar(const std::string &id) {
  return ImGui::BeginTabBar(id.c_str());
}

void GuiMetal::end_tab_bar() { ImGui::EndTabBar(); }

bool GuiMetal::begin_tab_item(const std::string &label) {
  return ImGui::BeginTabItem(label.c_str());
}

void GuiMetal::end_tab_item() { ImGui::EndTabItem(); }

bool GuiMetal::begin_table(const std::string &id, int columns) {
  return ImGui::BeginTable(id.c_str(), columns);
}

void GuiMetal::end_table() { ImGui::EndTable(); }

void GuiMetal::table_setup_column(const std::string &label) {
  ImGui::TableSetupColumn(label.c_str());
}

void GuiMetal::table_headers_row() { ImGui::TableHeadersRow(); }

void GuiMetal::table_next_row() { ImGui::TableNextRow(); }

bool GuiMetal::table_next_column() { return ImGui::TableNextColumn(); }

void GuiMetal::draw(taichi::lang::CommandList *cmd_list) {
  ImGui_ImplMetal_NewFrame(current_rpd_);

  // Rendering
  ImGui::Render();

  @autoreleasepool {
    MTLCommandBuffer_id buffer =
        static_cast<MetalCommandList *>(cmd_list)->finalize();

    MTLRenderCommandEncoder_id rce =
        [buffer renderCommandEncoderWithDescriptor:current_rpd_];
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), buffer, rce);
    [rce endEncoding];
  }
}
void GuiMetal::cleanup_render_resources() {
  ImGui_ImplMetal_Shutdown();
  current_rpd_ = nullptr;
}

GuiMetal::~GuiMetal() {
  if (app_context_->config.show_window) {
    ImGui_ImplGlfw_Shutdown();
  }
  cleanup_render_resources();
  ImGui::DestroyContext(imgui_context_);
}

bool GuiMetal::is_empty() { return is_empty_; }

} // namespace vulkan

} // namespace taichi::ui
