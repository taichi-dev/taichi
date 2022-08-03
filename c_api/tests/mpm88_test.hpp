#pragma once

#include <memory>
#include <vector>
#include "taichi/taichi_core.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VulkanDeviceCreator;

}
}  // namespace lang
}  // namespace taichi

namespace demo {

class MPM88DemoImpl;
class MPM88Demo {
 public:
  MPM88Demo(const std::string &aot_path, TiArch arch);
  ~MPM88Demo();

  void Step();

 private:
  std::unique_ptr<MPM88DemoImpl> impl_{nullptr};
};
}  // namespace demo
