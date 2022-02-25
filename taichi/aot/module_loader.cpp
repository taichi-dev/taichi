#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {

AotKernel *AotModuleLoader::get_kernel(const std::string &name) {
  auto itr = loaded_kernels_.find(name);
  if (itr != loaded_kernels_.end()) {
    return itr->second.get();
  }
  auto k = make_new_kernel(name);
  auto *kptr = k.get();
  kernel_holders_.push_back(std::move(k));
  loaded_kernels_[name] = kptr;
  return kptr;
}

}  // namespace lang
}  // namespace taichi
