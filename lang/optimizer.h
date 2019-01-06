#include "program.h"

TLANG_NAMESPACE_BEGIN

class Optimizer {
 public:
  virtual void run(Expr &expr) {
    while (search_and_replace(expr))
      ;
  }

  bool search_and_replace(Expr &expr);
};

TLANG_NAMESPACE_END
