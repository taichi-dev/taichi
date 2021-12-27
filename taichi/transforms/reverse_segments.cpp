#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/system/profiler.h"

#include <set>
#include <stack>

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

void mixed_statement_checker(Block *block) {
  bool has_for;
  bool has_non_for;
  std::stack<Block *> for_loops_stmts;
  for_loops_stmts.push(block);
  while (!for_loops_stmts.empty()) {
    has_for = false;
    has_non_for = false;
    auto sub_block = for_loops_stmts.top();
    for_loops_stmts.pop();
    int for_loops_num = 0;
    for (auto &&s : sub_block->statements) {
      if (s->is<FrontendForStmt>()) {
        has_for = true;
        for_loops_num++;
        for_loops_stmts.push(
            static_cast<FrontendForStmt *>(s.get())->body.get());
      } else {
        has_non_for = true;
      }
      if (for_loops_num >= 2) {
        TI_ERROR(
            "Invalid program input for autodiff: "
            "The outer for-loop contains more than one for-loops. \n"
            "Please check the documentation "
            "for the \"Kernel Simplicity Rule\" \"differentiable_task3\":\n"
            "https://docs.taichi.graphics/lang/articles/advanced/"
            "differentiable_programming#kernel-simplicity-rule");
      }
    }
    if (has_for && has_non_for)
      TI_ERROR(
          "Invalid program input for autodiff: "
          "Mixed usage of for-loop and a statement without looping. \n"
          "Please check the documentation "
          "for the \"Kernel Simplicity Rule\" \"differentiable_task4\":\n"
          "https://docs.taichi.graphics/lang/articles/advanced/"
          "differentiable_programming#kernel-simplicity-rule");
  }
}

void reverse_segments(IRNode *root) {
  TI_AUTO_PROF;
  auto block = dynamic_cast<Block *>(root);
  std::vector<std::vector<pStmt>> statement_blocks(1);
  bool has_for = false;
  bool has_non_for = false;
  for (auto &&s : block->statements) {
    if (s->is<FrontendForStmt>()) {
      has_for = true;
      mixed_statement_checker(
          static_cast<FrontendForStmt *>(s.get())->body.get());
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
        "Mixed usage of for-loop and a statement without looping. \n"
        "Please check the documentation "
        "for the \"Kernel Simplicity Rule\" \"differentiable_task4\":\n"
        "https://docs.taichi.graphics/lang/articles/advanced/"
        "differentiable_programming#kernel-simplicity-rule");
  }
  for (auto &sblock : statement_blocks) {
    for (auto &&s : sblock) {
      block->statements.push_back(std::move(s));
    }
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
