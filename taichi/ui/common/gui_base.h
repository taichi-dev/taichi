#pragma once
#include <string>
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
  virtual float slider_float(const std::string &name,
                             float old_value,
                             float minimum,
                             float maximum) = 0;
  virtual glm::vec3 color_edit_3(const std::string &name,
                                 glm::vec3 old_value) = 0;
  virtual bool button(const std::string &text) = 0;
  virtual void prepare_for_next_frame() = 0;
  virtual ~GuiBase() = default;
};

}  // namespace taichi::ui
