#include "taichi/ir/expression.h"

namespace taichi {
namespace lang {

std::string Expression::get_attribute(const std::string &key) const {
  if (auto it = attributes.find(key); it == attributes.end()) {
    TI_ERROR("Attribute {} not found.", key);
  } else {
    return it->second;
  }
}

}  // namespace lang
}  // namespace taichi
