#pragma once
#include <string>
#include <vector>
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

class GuiBase {
 public:
  virtual void begin(const std::string &name,
                     float x,
                     float y,
                     float width,
                     float height) = 0;
  virtual void end() = 0;
  virtual void text(const std::string &text) = 0;
  virtual void text(const std::string &text, glm::vec3 color) = 0;
  virtual bool checkbox(const std::string &name, bool old_value) = 0;
  virtual int slider_int(const std::string &name,
                         int old_value,
                         int minimum,
                         int maximum) = 0;
  virtual glm::ivec2 slider_int2(const std::string &name,
                                 glm::ivec2 old_value,
                                 int minimum,
                                 int maximum) = 0;
  virtual glm::ivec3 slider_int3(const std::string &name,
                                 glm::ivec3 old_value,
                                 int minimum,
                                 int maximum) = 0;
  virtual glm::ivec4 slider_int4(const std::string &name,
                                 glm::ivec4 old_value,
                                 int minimum,
                                 int maximum) = 0;
  virtual float slider_float(const std::string &name,
                             float old_value,
                             float minimum,
                             float maximum) = 0;
  virtual glm::vec2 slider_float2(const std::string &name,
                                  glm::vec2 old_value,
                                  float minimum,
                                  float maximum) = 0;
  virtual glm::vec3 slider_float3(const std::string &name,
                                  glm::vec3 old_value,
                                  float minimum,
                                  float maximum) = 0;
  virtual glm::vec4 slider_float4(const std::string &name,
                                  glm::vec4 old_value,
                                  float minimum,
                                  float maximum) = 0;
  virtual glm::vec3 color_edit_3(const std::string &name,
                                 glm::vec3 old_value) = 0;
  virtual glm::vec4 color_edit_4(const std::string &name,
                                 glm::vec4 old_value) = 0;
  virtual glm::vec3 color_picker_3(const std::string &name,
                                   glm::vec3 old_value) = 0;
  virtual glm::vec4 color_picker_4(const std::string &name,
                                   glm::vec4 old_value) = 0;
  virtual bool button(const std::string &text) = 0;
  virtual int combo(const std::string &label,
                    int current_item,
                    const std::vector<const char *> &items) = 0;
  virtual int input_int(const std::string &label, int old_value) = 0;
  virtual glm::ivec2 input_int2(const std::string &label,
                                glm::ivec2 old_value) = 0;
  virtual glm::ivec3 input_int3(const std::string &label,
                                glm::ivec3 old_value) = 0;
  virtual glm::ivec4 input_int4(const std::string &label,
                                glm::ivec4 old_value) = 0;
  virtual float input_float(const std::string &label, float old_value) = 0;
  virtual glm::vec2 input_float2(const std::string &label,
                                 glm::vec2 old_value) = 0;
  virtual glm::vec3 input_float3(const std::string &label,
                                 glm::vec3 old_value) = 0;
  virtual glm::vec4 input_float4(const std::string &label,
                                 glm::vec4 old_value) = 0;
  virtual int drag_int(const std::string &label,
                       int old_value,
                       float speed,
                       int minimum,
                       int maximum) = 0;
  virtual glm::ivec2 drag_int2(const std::string &label,
                               glm::ivec2 old_value,
                               float speed,
                               int minimum,
                               int maximum) = 0;
  virtual glm::ivec3 drag_int3(const std::string &label,
                               glm::ivec3 old_value,
                               float speed,
                               int minimum,
                               int maximum) = 0;
  virtual glm::ivec4 drag_int4(const std::string &label,
                               glm::ivec4 old_value,
                               float speed,
                               int minimum,
                               int maximum) = 0;
  virtual float drag_float(const std::string &label,
                           float old_value,
                           float speed,
                           float minimum,
                           float maximum) = 0;
  virtual glm::vec2 drag_float2(const std::string &label,
                                glm::vec2 old_value,
                                float speed,
                                float minimum,
                                float maximum) = 0;
  virtual glm::vec3 drag_float3(const std::string &label,
                                glm::vec3 old_value,
                                float speed,
                                float minimum,
                                float maximum) = 0;
  virtual glm::vec4 drag_float4(const std::string &label,
                                glm::vec4 old_value,
                                float speed,
                                float minimum,
                                float maximum) = 0;
  virtual bool tree_node_push(const std::string &label) = 0;
  virtual void tree_node_pop() = 0;
  virtual void separator() = 0;
  virtual void same_line() = 0;
  virtual void indent() = 0;
  virtual void unindent() = 0;
  virtual void progress_bar(float fraction) = 0;
  virtual bool collapsing_header(const std::string &label) = 0;
  virtual bool selectable(const std::string &label, bool selected) = 0;
  virtual bool radio_button(const std::string &label, bool active) = 0;
  virtual int listbox(const std::string &label,
                      int current_item,
                      const std::vector<const char *> &items,
                      int height_in_items) = 0;
  virtual bool begin_tab_bar(const std::string &id) = 0;
  virtual void end_tab_bar() = 0;
  virtual bool begin_tab_item(const std::string &label) = 0;
  virtual void end_tab_item() = 0;
  virtual bool begin_table(const std::string &id, int columns) = 0;
  virtual void end_table() = 0;
  virtual void table_setup_column(const std::string &label) = 0;
  virtual void table_headers_row() = 0;
  virtual void table_next_row() = 0;
  virtual bool table_next_column() = 0;
  virtual void prepare_for_next_frame() = 0;
  virtual ~GuiBase() = default;
};

}  // namespace taichi::ui
