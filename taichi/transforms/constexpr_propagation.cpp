#include "taichi/ir/transforms.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

#include <unordered_set>
#include <functional>

TLANG_NAMESPACE_BEGIN

namespace {

// A statement is const expr in Taichi IR, iff both its value and the control
// flow reaching it is constant w.r.t. const_seed(...)

class ConstExprPropagation : public IRVisitor {
 public:
  using is_const_seed_func = std::function<bool(Stmt *)>;

  ConstExprPropagation(const is_const_seed_func &is_const_seed)
      : is_const_seed_(is_const_seed) {
    allow_undefined_visitor = true;
  }

  bool is_const(Stmt *stmt) {
    if (is_const_seed_(stmt)) {
      return true;
    } else {
      return const_stmts_.find(stmt) != const_stmts_.end();
    }
  };

  void visit(UnaryOpStmt *stmt) override {
    if (is_const(stmt->operand)) {
      const_stmts_.insert(stmt);
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    if (is_const(stmt->lhs) && is_const(stmt->rhs)) {
      const_stmts_.insert(stmt);
    }
  }

  void visit(IfStmt *stmt) override {
    if (is_const(stmt->cond)) {
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

namespace irpass {
std::unordered_set<Stmt *> constexpr_prop(
    Block *block,
    std::function<bool(Stmt *)> is_const_seed) {
  return ConstExprPropagation::run(block, is_const_seed);
}
}  // namespace irpass

TLANG_NAMESPACE_END
