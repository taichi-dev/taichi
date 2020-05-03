#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expr.h"

TLANG_NAMESPACE_BEGIN

class FrontendAllocaStmt : public Stmt {
 public:
  Identifier ident;

  FrontendAllocaStmt(const Identifier &lhs, DataType type) : ident(lhs) {
    ret_type = VectorType(1, type);
  }

  DEFINE_ACCEPT
};

TLANG_NAMESPACE_END
