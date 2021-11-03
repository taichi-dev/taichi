#pragma once

#include <string>
#include "taichi/ui/utils/utils.h"
#include "taichi/program/arch.h"

TI_UI_NAMESPACE_BEGIN

struct AppConfig {
  std::string name;
  int width{0};
  int height{0};
  bool vsync{false};
  bool show_window{true};
  std::string package_path;
  taichi::lang::Arch ti_arch;
  bool is_packed_mode{false};
};

TI_UI_NAMESPACE_END
