#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

class Visitor {
public:
  enum class Order { parent_first, child_first };

  Order order;

  Visitor(Order order = Order::parent_first) : order(order) {
  }

  virtual void visit(Expr &expr) = 0;
};

}

TC_NAMESPACE_END
