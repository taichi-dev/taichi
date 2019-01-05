#pragma once

#include "expr.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class LoopVectorizer {
 public:
  int factor;
  int vectorized_id;
  std::map<Expr, Expr> input_to_vectorized;

  LoopVectorizer() {
    vectorized_id = -1;
  }

  Expr run(Expr &ret,
           SNode *snode,
           int factor);  // modify the kernel to be loop-vectorized

  Expr vectorize(Expr expr);
};

TLANG_NAMESPACE_END
