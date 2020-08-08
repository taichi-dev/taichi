// This pass analyzes compile-time known offsets for two values.

#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

DiffRange operator+(const DiffRange &a, const DiffRange &b) {
  return DiffRange(a.related_() && b.related_(), a.coeff + b.coeff,
                   a.low + b.low, a.high + b.high - 1);
}

DiffRange operator-(const DiffRange &a, const DiffRange &b) {
  return DiffRange(a.related_() && b.related_(), a.coeff - b.coeff,
                   a.low - b.high + 1, a.high - b.low);
}

class ValueDiffLoopIndex : public IRVisitor {
 public:
  // first: related, second: offset
  using ret_type = DiffRange;
  int lane;  // Note:  lane may change when visiting ElementShuffle
  Stmt *input_stmt, *loop;
  int loop_index;
  std::map<int, ret_type> results;

  ValueDiffLoopIndex(Stmt *stmt, int lane, Stmt *loop, int loop_index)
      : lane(lane), input_stmt(stmt), loop(loop), loop_index(loop_index) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    results[stmt->instance_id] = DiffRange();
  }

  void visit(GlobalLoadStmt *stmt) override {
    results[stmt->instance_id] = DiffRange();
  }

  void visit(LoopIndexStmt *stmt) override {
    results[stmt->instance_id] =
        DiffRange(stmt->loop == loop && stmt->index == loop_index, 1, 0);
  }

  void visit(ElementShuffleStmt *stmt) override {
    int old_lane = lane;
    TI_ASSERT(stmt->width() == 1);
    auto src = stmt->elements[lane].stmt;
    lane = stmt->elements[lane].index;
    src->accept(this);
    results[stmt->instance_id] = results[src->instance_id];
    lane = old_lane;
  }

  void visit(ConstStmt *stmt) override {
    if (stmt->val[lane].dt == DataType::i32) {
      results[stmt->instance_id] = DiffRange(true, 0, stmt->val[lane].val_i32);
    } else {
      results[stmt->instance_id] = DiffRange();
    }
  }

  void visit(RangeAssumptionStmt *stmt) override {
    stmt->base->accept(this);
    results[stmt->instance_id] = results[stmt->base->instance_id] +
                                 DiffRange(true, 0, stmt->low, stmt->high);
  }

  void visit(BinaryOpStmt *stmt) override {
    if (stmt->op_type == BinaryOpType::add ||
        stmt->op_type == BinaryOpType::sub) {
      stmt->lhs->accept(this);
      stmt->rhs->accept(this);
      auto ret1 = results[stmt->lhs->instance_id];
      auto ret2 = results[stmt->rhs->instance_id];
      if (ret1.related_() && ret2.related_()) {
        if (stmt->op_type == BinaryOpType::add) {
          results[stmt->instance_id] = ret1 + ret2;
        } else {
          results[stmt->instance_id] = ret1 - ret2;
        }
        return;
      }
    }
    results[stmt->instance_id] = {false, 0};
  }

  ret_type run() {
    input_stmt->accept(this);
    return results[input_stmt->instance_id];
  }
};

class FindDirectValueBaseAndOffset : public IRVisitor {
 public:
  // In the return value, <first> is true if this class finds that the input
  // statement has value equal to <second> + <third> (base + offset), or
  // <first> is false if this class can't find the decomposition.
  using ret_type = std::tuple<bool, Stmt *, int>;
  ret_type result;
  FindDirectValueBaseAndOffset() : result(false, nullptr, 0) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    result = std::make_tuple(false, nullptr, 0);
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    if (stmt->val[0].dt == DataType::i32) {
      result = std::make_tuple(true, nullptr, stmt->val[0].val_i32);
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    if (stmt->rhs->is<ConstStmt>())
      stmt->rhs->accept(this);
    if (!std::get<0>(result) || std::get<1>(result) != nullptr ||
        (stmt->op_type != BinaryOpType::add &&
         stmt->op_type != BinaryOpType::sub)) {
      result = std::make_tuple(false, nullptr, 0);
      return;
    }
    if (stmt->op_type == BinaryOpType::sub)
      std::get<2>(result) = -std::get<2>(result);
    std::get<1>(result) = stmt->lhs;
  }

  static ret_type run(Stmt *val) {
    FindDirectValueBaseAndOffset instance;
    val->accept(&instance);
    return instance.result;
  }
};

namespace irpass::analysis {

DiffRange value_diff_loop_index(Stmt *stmt, Stmt *loop, int index_id) {
  TI_ASSERT(loop->is<StructForStmt>() || loop->is<OffloadedStmt>());
  if (loop->is<OffloadedStmt>()) {
    TI_ASSERT(loop->as<OffloadedStmt>()->task_type ==
              OffloadedStmt::TaskType::struct_for);
  }
  if (auto loop_index = stmt->cast<LoopIndexStmt>(); loop_index) {
    if (loop_index->loop == loop && loop_index->index == index_id) {
      return DiffRange(true, 1, 0);
    }
  }
  TI_ASSERT(stmt->width() == 1);
  auto diff = ValueDiffLoopIndex(stmt, 0, loop, index_id);
  return diff.run();
}

std::pair<bool, int> value_diff_ptr_index(Stmt *val1, Stmt *val2) {
  // <first>: whether the difference of the value of two statements is certain.
  // <second>: the difference of the value of two statements (i.e. val1 - val2)
  // if <first> is true.
  if (val1 == val2)
    return std::make_pair(true, 0);
  auto v1 = FindDirectValueBaseAndOffset::run(val1);
  auto v2 = FindDirectValueBaseAndOffset::run(val2);
  if (!std::get<0>(v1) || !std::get<0>(v2) ||
      std::get<1>(v1) != std::get<1>(v2)) {
    // uncertain
    return std::make_pair(false, 0);
  }
  return std::make_pair(true, std::get<2>(v1) - std::get<2>(v2));
}

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
