#pragma once

#include <memory>
#include "taichi/gui/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"
#include <vector>

namespace demo {

class MPM88DemoImpl;
class MPM88Demo {
 public:
  MPM88Demo(const std::string &aot_path, const std::string &arch_name);
  ~MPM88Demo();

  void Step();

 private:
  std::unique_ptr<MPM88DemoImpl> impl_{nullptr};
  std::shared_ptr<taichi::ui::vulkan::Gui> gui_{nullptr};
  std::unique_ptr<taichi::ui::vulkan::Renderer> renderer{nullptr};
  GLFWwindow *window{nullptr};
  taichi::ui::FieldInfo f_info;
  taichi::ui::CirclesInfo circles;
};
}  // namespace demo
