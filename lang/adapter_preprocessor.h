#pragma once

#include "expr.h"
#include "visitor.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

class AdapterPreprocessor : public Visitor {
 public:
  Kernel *kernel;
  int group_size;

  AdapterPreprocessor() : Visitor(Visitor::Order::parent_first) {
  }

  void run(Kernel &kernel, Expr &expr, int group_size) {
    this->kernel = &kernel;
    this->group_size = group_size;
    expr.accept(*this);
  }

  void visit(Expr &expr) override {
    if (expr->type == NodeType::adapter_load) {
      // generate offsets in the linearized input adapter.
      // (There may be multiple VV's)
      // (This is invariant even if we do vector splitting.)
      auto &ad = kernel->adapter(expr[0]->value<int>());
      std::vector<int> offsets_val;
      TC_ASSERT(expr->lanes == kernel->parallel_instances * group_size);
      for (int i = 0; i < kernel->parallel_instances; i++) {
        for (int j = 0; j < group_size; j++) {
          int &elem_id = expr[1]->attribute<int>(0, i * group_size + j);
          elem_id = i * ad.input_group_size +
                    elem_id / ad.input_group_size * ad.input_group_size *
                        kernel->parallel_instances +
                    elem_id % ad.input_group_size;
          expr[1]->attribute<int>(1, i * group_size + j) =
              1;  // marked as preprocessed
        }
      }
    }
  }
};

TLANG_NAMESPACE_END
