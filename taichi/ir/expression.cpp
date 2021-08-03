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

ExprGroup ExprGroup::loaded() const {
  auto indices_loaded = *this;
  for (int i = 0; i < (int)this->size(); i++)
    indices_loaded[i].set(load_if_ptr(indices_loaded[i]));
  return indices_loaded;
}

std::string ExprGroup::serialize() const {
  std::string ret;
  for (int i = 0; i < (int)exprs.size(); i++) {
    ret += exprs[i].serialize();
    if (i + 1 < (int)exprs.size()) {
      ret += ", ";
    }
  }
  return ret;
}

}  // namespace lang
}  // namespace taichi
