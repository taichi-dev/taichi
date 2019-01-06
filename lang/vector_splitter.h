#pragma once

#include "expr.h"
#include "visitor.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class VectorSplitter : public Visitor {
 public:
  int num_splits;
  int target_lanes;
  Kernel *kernel;

  std::map<Expr, std::vector<Expr>> split;

  VectorSplitter(int target_lanes)
      : Visitor(Visitor::Order::child_first), target_lanes(target_lanes) {
  }

  void run(Expr &expr);

  void visit(Expr &expr) override;
};

TLANG_NAMESPACE_END
