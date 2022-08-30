// This pass analyzes compile-time known offsets for two values.

#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"

namespace taichi {
namespace lang {

DiffRange operator+(const DiffRange &a, const DiffRange &b) {
  return DiffRange(a.related() && b.related(), a.coeff + b.coeff, a.low + b.low,
                   a.high + b.high - 1);
}

DiffRange operator-(const DiffRange &a, const DiffRange &b) {
  return DiffRange(a.related() && b.related(), a.coeff - b.coeff,
                   a.low - b.high + 1, a.high - b.low);
}

DiffRange operator*(const DiffRange &a, const DiffRange &b) {
  return DiffRange(
      a.related() && b.related() && a.coeff * b.coeff == 0,
      fmax(a.low * b.coeff, a.coeff * b.low),
      fmin(a.low * b.low,
           fmin(a.low * (b.high - 1),
                fmin(b.low * (a.high - 1), (a.high - 1) * (b.high - 1)))),
      fmax(a.low * b.low,
           fmax(a.low * (b.high - 1),
                fmax(b.low * (a.high - 1), (a.high - 1) * (b.high - 1)))) +
          1);
}

DiffRange operator<<(const DiffRange &a, const DiffRange &b) {
  return DiffRange(
      a.related() && b.related() && b.coeff == 0 && b.high - b.low == 1,
      a.coeff << b.low, a.low << b.low, ((a.high - 1) << b.low) + 1);
}

namespace {

class ValueDiffLoopIndex : public IRVisitor {
 public:
  // first: related, second: offset
  using ret_type = DiffRange;
  Stmt *input_stmt, *loop;
  int loop_index;
  std::map<int, ret_type> results;

  ValueDiffLoopIndex(Stmt *stmt, Stmt *loop, int loop_index)
      : input_stmt(stmt), loop(loop), loop_index(loop_index) {
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
    results[stmt->instance_id] = DiffRange();
    if (stmt->loop == loop && stmt->index == loop_index) {
      results[stmt->instance_id] =
          DiffRange(/*related=*/true, /*coeff=*/1, /*low=*/0);
    } else if (auto range_for = stmt->loop->cast<RangeForStmt>()) {
      if (range_for->begin->is<ConstStmt>() &&
          range_for->end->is<ConstStmt>()) {
        auto begin_val = range_for->begin->as<ConstStmt>()->val.val_int();
        auto end_val = range_for->end->as<ConstStmt>()->val.val_int();
        // We have begin_val <= end_val even when range_for->reversed is true:
        // in that case, the loop is iterated from end_val - 1 to begin_val.
        results[stmt->instance_id] = DiffRange(
            /*related=*/true, /*coeff=*/0, /*low=*/begin_val, /*high=*/end_val);
      }
    }
  }

  void visit(ConstStmt *stmt) override {
    if (stmt->val.dt->is_primitive(PrimitiveTypeID::i32)) {
      results[stmt->instance_id] = DiffRange(true, 0, stmt->val.val_i32);
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
        stmt->op_type == BinaryOpType::sub ||
        stmt->op_type == BinaryOpType::mul ||
        stmt->op_type == BinaryOpType::bit_shl) {
      stmt->lhs->accept(this);
      stmt->rhs->accept(this);
      auto ret1 = results[stmt->lhs->instance_id];
      auto ret2 = results[stmt->rhs->instance_id];
      if (ret1.related() && ret2.related()) {
        if (stmt->op_type == BinaryOpType::add) {
          results[stmt->instance_id] = ret1 + ret2;
        } else if (stmt->op_type == BinaryOpType::sub) {
          results[stmt->instance_id] = ret1 - ret2;
        } else if (stmt->op_type == BinaryOpType::mul) {
          results[stmt->instance_id] = ret1 * ret2;
        } else {
          results[stmt->instance_id] = ret1 << ret2;
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
    if (stmt->val.dt->is_primitive(PrimitiveTypeID::i32)) {
      result = std::make_tuple(true, nullptr, stmt->val.val_i32);
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

}  // namespace

namespace irpass {
namespace analysis {

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
  auto diff = ValueDiffLoopIndex(stmt, loop, index_id);
  return diff.run();
}

DiffPtrResult value_diff_ptr_index(Stmt *val1, Stmt *val2) {
  if (val1 == val2) {
    return DiffPtrResult::make_certain(0);
  }
  auto v1 = FindDirectValueBaseAndOffset::run(val1);
  auto v2 = FindDirectValueBaseAndOffset::run(val2);
  if (!std::get<0>(v1) || !std::get<0>(v2) ||
      std::get<1>(v1) != std::get<1>(v2)) {
    return DiffPtrResult::make_uncertain();
  }
  return DiffPtrResult::make_certain(std::get<2>(v1) - std::get<2>(v2));
}

}  // namespace analysis
}  // namespace irpass
}  // namespace lang
}  // namespace taichi
