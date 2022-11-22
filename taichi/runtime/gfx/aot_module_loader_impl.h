#pragma once

#include <any>
#include <string>
#include <vector>

#include "taichi/runtime/gfx/aot_utils.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_builder.h"
#include "taichi/aot/module_loader.h"

namespace taichi::lang {
namespace gfx {

struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path;
  GfxRuntime *runtime{nullptr};
  bool enable_lazy_loading{false};

  AotModuleParams() = default;

  AotModuleParams(const std::string &path, GfxRuntime *rt)
      : module_path(path), runtime(rt) {
  }
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(
    std::any mod_params,
    Arch device_api_backend);

}  // namespace gfx
}  // namespace taichi::lang
