#include "taichi/runtime/runtime.h"

namespace taichi::lang {

std::unique_ptr<Runtime> Runtime::create(Arch arch) {
  auto &factories = get_factories();
  if (auto factory = factories.find(arch); factory != factories.end()) {
    return factory->second();
  } else {
    return nullptr;
  }
}

}  // namespace taichi::lang
