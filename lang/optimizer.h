#include "program.h"
#include <set>

TLANG_NAMESPACE_BEGIN

class Optimizer {
 protected:
  Kernel *kernel;
  Expr root;

 public:
  void run(Kernel &ker, Expr &expr) {
    root = expr;
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

  void replace(Expr a, Expr b) {
    std::set<Expr> visited;
    std::function<bool(Expr &)> visit = [&](Expr &v) -> bool {
      if (visited.find(v) != visited.end())
        return false;
      visited.insert(v);
      for (int i = 0; i < (int)v->ch.size(); i++) {
        if (v->ch[i] == a) {
          v->ch[i] = b;
          return true;
        } else if (visit(v->ch[i])) {
          return true;
        }
      }
      return false;
    };
    int c = 0;
    while (1) {
      visited.clear();
      if (!visit(root)) {
        break;
      }
      c++;
    }
    if (c > 1) {
      TC_INFO("Replacing {} Exprs altogether", c);
    }
  }
};

void apply_optimizers(Kernel &ker, Expr &expr);

TLANG_NAMESPACE_END
