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
  std::string package_path;
  taichi::lang::Arch ti_arch;
};

TI_UI_NAMESPACE_END
