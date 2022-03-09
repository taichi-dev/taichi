#pragma once

#include <string>
#include <vector>

#include "taichi/backends/vulkan/aot_utils.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VkRuntime;

struct AotModuleParams {
  std::string module_path;
  VkRuntime *runtime{nullptr};
};

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params);
}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
