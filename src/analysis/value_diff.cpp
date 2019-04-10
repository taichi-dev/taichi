#include "../ir.h"

TLANG_NAMESPACE_BEGIN

DiffRange operator+(const DiffRange &a, const DiffRange &b) {
  return DiffRange(a.related && b.related, a.low + b.low, a.high + b.high - 1);
}

DiffRange operator-(const DiffRange &a, const DiffRange &b) {
  return DiffRange(a.related && b.related, a.low - b.high + 1, a.high - b.low);
}

class ValueDiff : public IRVisitor {
 public:
  // first: related, second: offset
  using ret_type = DiffRange;
  int lane;
  Stmt *input_stmt, *alloc;
  std::map<int, ret_type> results;

  ValueDiff(Stmt *stmt, int lane, Stmt *alloc)
      : lane(lane), input_stmt(stmt), alloc(alloc) {
  }

  void visit(GlobalLoadStmt *stmt) override {
    results[stmt->instance_id] = DiffRange(false);
  }

  void visit(LocalLoadStmt *stmt) override {
    if (stmt->ptr[lane].var == alloc) {
      results[stmt->instance_id] = DiffRange(true, 0);
    }
  }

  void visit(ConstStmt *stmt) override {
    if (stmt->val[lane].dt == DataType::i32) {
      results[stmt->instance_id] = DiffRange(true, stmt->val[lane].val_i32);
    } else {
      results[stmt->instance_id] = DiffRange(false);
    }
  }

  void visit(RangeAssumptionStmt *stmt) override {
    stmt->base->accept(this);
    results[stmt->instance_id] = results[stmt->base->instance_id] +
                                 DiffRange(true, stmt->low, stmt->high);
  }

  void visit(BinaryOpStmt *stmt) override {
    if (stmt->op_type == BinaryType::add || stmt->op_type == BinaryType::sub) {
      if (stmt->lhs->is<LocalLoadStmt>() && stmt->rhs->is<ConstStmt>()) {
        stmt->lhs->accept(this);
        stmt->rhs->accept(this);
        auto ret1 = results[stmt->lhs->instance_id];
        auto ret2 = results[stmt->rhs->instance_id];
        if (ret1.related && ret2.related) {
          if (stmt->op_type == BinaryType::add) {
            results[stmt->instance_id] = ret1 + ret2;
          } else {
            results[stmt->instance_id] = ret1 - ret2;
          }
          return;
        }
      }
    }
    results[stmt->instance_id] = {false, 0};
  }

  ret_type run() {
    input_stmt->accept(this);
    return results[input_stmt->instance_id];
  }
};

namespace analysis {

DiffRange value_diff(Stmt *stmt, int lane, Stmt *alloca) {
  ValueDiff _(stmt, lane, alloca);
  return _.run();
}

}  // namespace analysis

TLANG_NAMESPACE_END
