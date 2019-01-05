#pragma once

#include "expr.h"
#include "visitor.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class SLPVectorizer : public Visitor {
 public:
  std::map<Expr, Expr> scalar_to_vector;
  int group_size;

  SLPVectorizer() : Visitor(Visitor::Order::parent_first) {
  }

  void sort(Expr &expr);

  void run(Kernel &kernel, int group_size);

  void visit(Expr &expr) override;
};

TLANG_NAMESPACE_END
