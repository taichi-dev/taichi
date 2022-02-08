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

void ExprGroup::serialize(std::ostream &ss) const {
  for (int i = 0; i < (int)exprs.size(); i++) {
    exprs[i].serialize(ss);
    if (i + 1 < (int)exprs.size()) {
      ss << ", ";
    }
  }
}

std::string ExprGroup::serialize() const {
  std::stringstream ss;
  serialize(ss);
  return ss.str();
}

}  // namespace lang
}  // namespace taichi
