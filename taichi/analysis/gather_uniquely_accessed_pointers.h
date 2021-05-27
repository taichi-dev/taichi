#pragma once

#include "taichi/ir/pass.h"

namespace taichi {
namespace lang {

class GatherUniquelyAccessedBitStructsPass : public Pass {
 public:
  static const PassID id;

  struct Result {
    std::unordered_map<OffloadedStmt *,
                       std::unordered_map<const SNode *, GlobalPtrStmt *>>
        uniquely_accessed_bit_structs;
  };
};

}  // namespace lang
}  // namespace taichi
