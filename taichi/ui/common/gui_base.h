#pragma once
#include <string>
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

class GuiBase {
 public:
  virtual void begin(std::string name,
                     float x,
                     float y,
                     float width,
                     float height) {
  }
  virtual void end() {
  }
  virtual void text(std::string text) {
  }
  virtual bool checkbox(std::string name, bool old_value) {
    return false;
  }
  virtual float slider_float(std::string name,
                             float old_value,
                             float minimum,
                             float maximum) {
    return 0.0;
  }
  virtual glm::vec3 color_edit_3(std::string name, glm::vec3 old_value) {
    return glm::vec3(0.f, 0.f, 0.f);
  }
  virtual bool button(std::string text) {
    return false;
  }
};

TI_UI_NAMESPACE_END
