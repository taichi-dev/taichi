#pragma once

#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

namespace Tlang {

class Expr;

class Visitor {
public:
  enum class Order { parent_first, child_first };

  Order order;

  Visitor(Order order) : order(order) {
  }

  virtual void visit(Expr &expr) = 0;
};

}

TC_NAMESPACE_END
