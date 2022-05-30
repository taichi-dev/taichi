#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/system/profiler.h"

#include <set>

TLANG_NAMESPACE_BEGIN

namespace irpass {

// TODO: gather Expr as well?
class GatherStmts : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::vector<Stmt *> stmts;

  GatherStmts() {
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    stmts.push_back(stmt);
  }
};

void reverse_segments(IRNode *root) {
  TI_AUTO_PROF;
  auto block = dynamic_cast<Block *>(root);
  std::vector<std::vector<pStmt>> statement_blocks(1);
  bool has_for = false;
  bool has_non_for = false;
  for (auto &&s : block->statements) {
    if (s->is<FrontendForStmt>()) {
      has_for = true;
      statement_blocks.emplace_back();
      statement_blocks.back().push_back(std::move(s));
      statement_blocks.emplace_back();
    } else {
      has_non_for = true;
      statement_blocks.back().push_back(std::move(s));
    }
  }
  block->statements.clear();
  std::reverse(statement_blocks.begin(), statement_blocks.end());
  /*
  for (auto &b : statement_blocks) {
    std::vector<Stmt *> stmts;
    for (auto &s : b) {
      GatherStmts gather;
      s->accept(&gather);
      stmts.insert(stmts.end(), gather.stmts.begin(), gather.stmts.end());
    }
    std::set<Stmt *> stmt_set(stmts.begin(), stmts.end());
    bool valid = true;
    for (auto s : stmts) {
      for (auto op : s->get_operands()) {
        if (stmt_set.find(op) == stmt_set.end()) {
          valid = false;
        }
      }
    }
  }
    */
  if (has_for && has_non_for) {
    TI_ERROR(
        "Invalid program input for autodiff: "
        "Mixed usage of for-loops and statements without looping. \n"
        "Please split them into two kernels "
        "and check the documentation for more details:\n"
        "https://docs.taichi-lang.org/docs/"
        "differentiable_programming");
  }
  for (auto &sblock : statement_blocks) {
    for (auto &&s : sblock) {
      block->statements.push_back(std::move(s));
    }
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
