#pragma once

#include <string>
#include "taichi/ui/utils/utils.h"
#include "taichi/backends/arch.h"

namespace taichi {
namespace ui {

struct AppConfig {
  std::string name;
  int width{0};
  int height{0};
  bool vsync{false};
  bool show_window{true};
  std::string package_path;
  Arch ti_arch;
  bool is_packed_mode{false};
};

}  // namespace ui
}  // namespace taichi
