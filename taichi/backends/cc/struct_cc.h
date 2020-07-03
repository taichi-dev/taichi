#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCLayout;

class CCLayoutGen {
  // Generate corresponding C Source Code for Taichi Structures
 public:
  CCLayoutGen(SNode *root) : root(root) {
  }

  std::unique_ptr<CCLayout> compile();

 private:
  SNode *root;
};

}  // namespace cccp
TLANG_NAMESPACE_END
