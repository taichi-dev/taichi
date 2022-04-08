#include "c_api/include/taichi/aot/module.h"

#include "taichi/aot/module_loader.h"

namespace {

#include "c_api/src/inc/aot_casts.inc.h"
#include "c_api/src/inc/runtime_casts.inc.h"

}  // namespace

Taichi_Kernel *taichi_get_kernel_from_aot_module(Taichi_AotModule *m,
                                                 const char *name) {
  auto *mod = cppcast(m);
  auto *k = mod->get_kernel(name);
  return reinterpret_cast<Taichi_Kernel *>(k);
}

size_t taichi_get_root_size_from_aot_module(Taichi_AotModule *m) {
  return cppcast(m)->get_root_size();
}
