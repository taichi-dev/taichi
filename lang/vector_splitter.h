#pragma once

#include "expr.h"
#include "visitor.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class VectorSplitter : public Visitor {
 public:
  int input_lanes;
  int num_splits;
  int target_lanes;
  Kernel *kernel;

  std::map<Expr, std::vector<Expr>> split;

  VectorSplitter() : Visitor(Visitor::Order::parent_first) {
  }

  Expr run(Kernel &kernel, Expr &expr, int target_lanes);

  void visit(Expr &expr) override;
};

TLANG_NAMESPACE_END
