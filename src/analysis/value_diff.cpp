#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class ValueDiff : public IRVisitor {
 public:
  // first: related, second: offset
  using ret_type = std::pair<bool, int>;
  int lane;
  Stmt *input_stmt, *alloc;
  std::map<int, ret_type> results;

  ValueDiff(Stmt *stmt, int lane, Stmt *alloc)
      : lane(lane), input_stmt(stmt), alloc(alloc) {
  }

  void visit(LocalLoadStmt *stmt) override {
    if (stmt->ptr[lane].var == alloc) {
      results[stmt->instance_id] = {true, 0};
    }
  }

  void visit(ConstStmt *stmt) override {
    if (stmt->val[lane].dt == DataType::i32) {
      results[stmt->instance_id] = {true, stmt->val[lane].val_i32};
    } else {
      results[stmt->instance_id] = {false, 0};
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    if (stmt->op_type == BinaryType::add || stmt->op_type == BinaryType::sub) {
      if (stmt->lhs->is<LocalLoadStmt>() && stmt->rhs->is<ConstStmt>()) {
        stmt->lhs->accept(this);
        stmt->rhs->accept(this);
        auto ret1 = results[stmt->lhs->instance_id];
        auto ret2 = results[stmt->rhs->instance_id];
        if (ret1.first && ret2.first) {
          if (stmt->op_type == BinaryType::add) {
            results[stmt->instance_id] = {true, ret1.second + ret2.second};
          } else {
            results[stmt->instance_id] = {true, ret1.second - ret2.second};
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

std::pair<bool, int> value_diff(Stmt *stmt, int lane, Stmt *alloca) {
  ValueDiff _(stmt, lane, alloca);
  return _.run();
}

}  // namespace analysis

TLANG_NAMESPACE_END
