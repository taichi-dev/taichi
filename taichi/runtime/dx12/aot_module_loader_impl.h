#pragma once

#include <any>
#include <string>
#include <vector>

#include "taichi/aot/module_builder.h"
#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {
namespace directx12 {

struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path;
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(
    std::any mod_params,
    Arch device_api_backend);

}  // namespace directx12
}  // namespace lang
}  // namespace taichi
