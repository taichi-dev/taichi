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
                     float height) = 0;
  virtual void end() = 0;
  virtual void text(std::string text) = 0;
  virtual bool checkbox(std::string name, bool old_value) = 0;
  virtual int slider_int(std::string name,
                         int old_value,
                         int minimum,
                         int maximum) = 0;
  virtual float slider_float(std::string name,
                             float old_value,
                             float minimum,
                             float maximum) = 0;
  virtual glm::vec3 color_edit_3(std::string name, glm::vec3 old_value) = 0;
  virtual bool button(std::string text) = 0;
  virtual ~GuiBase() = default;
};

TI_UI_NAMESPACE_END
