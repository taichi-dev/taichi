#include "program.h"
#include <set>

TLANG_NAMESPACE_BEGIN

class Optimizer {
 protected:
  Kernel *kernel;

 public:
  void run(Kernel &ker, Expr &expr) {
    kernel = &ker;
    while (true) {
      visited.clear();
      if (!search_and_replace(expr)) {
        break;
      }
    }
  }

  std::set<Expr> visited;

  bool search_and_replace(Expr &expr);

  virtual bool optimize(Expr &expr) = 0;
};

void apply_optimizers(Kernel &ker, Expr &expr);

TLANG_NAMESPACE_END
