#pragma once

#include "expr.h"
#include "visitor.h"

namespace taichi::Tlang {

class Vectorizer : public Visitor {
 public:
  std::map<Expr, Expr> scalar_to_vector;
  int group_size;

  Vectorizer() : Visitor(Visitor::Order::parent_first) {
  }

  void sort(Expr &expr);

  Expr run(Expr &expr, int group_size);

  void visit(Expr &expr) override;
};
}
