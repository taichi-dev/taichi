#include "c_api/include/taichi/aot_module.h"

#include "taichi/aot/module_loader.h"

namespace {

#include "c_api/src/inc/runtime_casts.inc.h"

}  // namespace

TaichiKernel *get_taichi_kernel(AotModule *m, const char *name) {
  auto *mod = cppcast(m);
  auto *k = mod->get_kernel(name);
  return reinterpret_cast<TaichiKernel *>(k);
}
