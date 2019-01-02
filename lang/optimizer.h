#include "program.h"

TLANG_NAMESPACE_BEGIN

class Optimizer {
 public:
  virtual void run(Kernel &kernel) {
    while (search_and_replace(kernel.ret))
      ;
  }

  bool search_and_replace(Expr &expr) {
    for (auto ch : expr->ch) {
      bool ret = search_and_replace(ch);
      if (ret)
        return true;
    }

    if (expr->type == NodeType::load) {
      auto &addr_node = expr._pointer()._address();
      bool all_same = true;
      for (int i = 0; i < addr_node->lanes; i++) {
        if (addr_node->new_addresses(i) != addr_node->new_addresses(0))
          all_same = false;
      }
      if (all_same) {

      }
    }

    return false;
  }
};

TLANG_NAMESPACE_END
