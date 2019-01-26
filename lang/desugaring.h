#pragma once

#include <set>
#include "expr.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class Desugaring {
 public:
  std::set<Expr> visited;

  Desugaring() {
  }

  Expr run(Expr &expr) {
    visit(expr);
    return expr;
  }

  void visit(Expr expr) {
    if (visited.find(expr) == visited.end()) {
      visited.insert(expr);
    } else {
      return;
    }

    for (int i = 0; i < expr->ch.size(); i++) {
      auto &ch = expr->ch[i];
      if (ch->type == NodeType::pointer) {
        // consider add a load..
        if (expr->type != NodeType::load && expr->type != NodeType::store &&
            expr->type != NodeType::reduce) {
          TC_INFO("Desugar: add load to ptr");
          ch.set(Expr::load(ch));
        }
      }
      visit(ch);
    }
  }
};

TLANG_NAMESPACE_END
