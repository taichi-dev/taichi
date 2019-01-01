#pragma once

#include "expr.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class LoopVectorizer {
 public:
  int factor;
  std::map<Expr, Expr> input_to_vectorized;

  LoopVectorizer() {
    factor = 8;  // AVX2 only for now
  }

  void run(Kernel &kernel);  // modify the kernel to be loop-vectorized

  Expr vectorize(Expr expr);
};

TLANG_NAMESPACE_END
