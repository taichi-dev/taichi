#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "taichi/aot/module_loader.h"

namespace taichi::lang {
namespace metal {

class KernelManager;

struct AotModuleParams {
  std::string module_path;
  KernelManager *runtime{nullptr};
};

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params);
}  // namespace metal
}  // namespace taichi::lang
