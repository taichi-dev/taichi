#pragma once

#include "taichi/aot/module_loader.h"

namespace taichi::lang {

class LlvmRuntimeExecutor;

namespace cuda {

struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path;
  LlvmRuntimeExecutor *executor_{nullptr};
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(std::any mod_params);

}  // namespace cuda
}  // namespace taichi::lang
