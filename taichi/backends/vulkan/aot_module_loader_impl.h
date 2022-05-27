#pragma once

#include <any>
#include <string>
#include <vector>

#include "taichi/backends/vulkan/aot_utils.h"
#include "taichi/runtime/vulkan/runtime.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_builder.h"
#include "taichi/aot/module_loader.h"
#include "taichi/backends/vulkan/aot_module_builder_impl.h"
#include "taichi/backends/vulkan/vulkan_graph_data.h"

namespace taichi {
namespace lang {
namespace vulkan {
struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path;
  VkRuntime *runtime{nullptr};
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(std::any mod_params);

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
