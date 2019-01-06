#include "program.h"

TLANG_NAMESPACE_BEGIN

class Optimizer {
 public:
  virtual void run(Expr &expr) {
    while (true) {
      visited.clear();
      if (!search_and_replace(expr)) {
        break;
      }
    }
  }

  std::set<Expr> visited;

  bool search_and_replace(Expr &expr);
};

TLANG_NAMESPACE_END
