#include "taichi/lang_util.h"

#include <mutex>

TLANG_NAMESPACE_BEGIN

class TypeFactory {
 public:
  Type *get_primitive_type(PrimitiveType::primitive_type id);

 private:
  std::unordered_map<PrimitiveType::primitive_type, std::unique_ptr<Type>>
      primitive_types_;

  std::mutex mut;
};

TLANG_NAMESPACE_END
