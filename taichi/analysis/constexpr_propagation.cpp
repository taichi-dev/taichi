#include "taichi/ir/visitors.h"
#include "taichi/ir/statements.h"

#include <unordered_set>
#include <functional>

TLANG_NAMESPACE_BEGIN

namespace {

// A statement is considered constexpr in this pass, iff both its value and the
// control flow reaching it are constant w.r.t. const_seed(...)

class ConstExprPropagation : public IRVisitor {
 public:
  using is_const_seed_func = std::function<bool(Stmt *)>;

  ConstExprPropagation(const is_const_seed_func &is_const_seed)
      : is_const_seed_(is_const_seed) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  bool generic_test(Stmt *stmt) {
    if (is_const_seed_(stmt)) {
      const_stmts_.insert(stmt);
      return true;
    } else {
      return false;
    }
  }

  void visit(Stmt *stmt) override {
    generic_test(stmt);
  }

  bool is_inferred_const(Stmt *stmt) {
    // Note: every statement that tests true by "is_const_seed_" should have
    // already been included in const_stmts_.
    return const_stmts_.find(stmt) != const_stmts_.end();
  };

  void visit(UnaryOpStmt *stmt) override {
    if (generic_test(stmt))
      return;
    if (is_inferred_const(stmt->operand)) {
      const_stmts_.insert(stmt);
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    if (generic_test(stmt))
      return;
    if (is_inferred_const(stmt->lhs) && is_inferred_const(stmt->rhs)) {
      const_stmts_.insert(stmt);
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    if (generic_test(stmt))
      return;
    if (is_inferred_const(stmt->op1) && is_inferred_const(stmt->op2) &&
        is_inferred_const(stmt->op3)) {
      const_stmts_.insert(stmt);
    }
  }

  void visit(IfStmt *stmt) override {
    // If the condition is constexpr, then the control flow is also considered
    // const.
    if (is_inferred_const(stmt->cond)) {
      if (stmt->true_statements)
        stmt->true_statements->accept(this);
      if (stmt->false_statements)
        stmt->false_statements->accept(this);
    }
  }

  void visit(Block *block) override {
    for (auto &stmt : block->statements)
      stmt->accept(this);
  }

  // TODO: how do we rigorously define constexpr in RangeFor loops?

  static std::unordered_set<Stmt *> run(
      Block *block,
      const std::function<bool(Stmt *)> &is_const_seed) {
    ConstExprPropagation prop(is_const_seed);
    block->accept(&prop);
    return prop.const_stmts_;
  }

 private:
  is_const_seed_func is_const_seed_;
  std::unordered_set<Stmt *> const_stmts_;
};

}  // namespace

namespace irpass::analysis {
std::unordered_set<Stmt *> constexpr_prop(
    Block *block,
    std::function<bool(Stmt *)> is_const_seed) {
  return ConstExprPropagation::run(block, is_const_seed);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
