#include "taichi/transforms/inlining.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

// Inline all functions.
class Inliner : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit Inliner() : BasicStmtVisitor() {
  }

  void visit(FuncCallStmt *stmt) override {
    auto *func = stmt->func;
    TI_ASSERT(func);
    TI_ASSERT(func->args.size() == stmt->args.size());
    TI_ASSERT(func->ir->is<Block>());
    TI_ASSERT(func->rets.size() <= 1);
    auto inlined_ir = irpass::analysis::clone(func->ir.get());
    if (!func->args.empty()) {
      irpass::replace_statements(
          inlined_ir.get(),
          /*filter=*/[&](Stmt *s) { return s->is<ArgLoadStmt>(); },
          /*finder=*/
          [&](Stmt *s) { return stmt->args[s->as<ArgLoadStmt>()->arg_id]; });
    }
    if (func->rets.empty()) {
      modifier_.replace_with(stmt,
                             std::move(inlined_ir->as<Block>()->statements));
    } else {
      if (irpass::analysis::gather_statements(inlined_ir.get(), [&](Stmt *s) {
            return s->is<ReturnStmt>();
          }).size() > 1) {
        TI_WARN(
            "Multiple returns in function \"{}\" may not be handled "
            "properly.\n{}",
            func->get_name(), stmt->tb);
      }
      // Use a local variable to store the return value
      auto *return_address = inlined_ir->as<Block>()->insert(
          Stmt::make<AllocaStmt>(func->rets[0].dt), /*location=*/0);
      irpass::replace_and_insert_statements(
          inlined_ir.get(),
          /*filter=*/[&](Stmt *s) { return s->is<ReturnStmt>(); },
          /*generator=*/
          [&](Stmt *s) {
            TI_ASSERT(s->as<ReturnStmt>()->values.size() == 1);
            return Stmt::make<LocalStoreStmt>(return_address,
                                              s->as<ReturnStmt>()->values[0]);
          });
      modifier_.insert_before(stmt,
                              std::move(inlined_ir->as<Block>()->statements));
      // Load the return value here
      modifier_.replace_with(
          stmt, Stmt::make<LocalLoadStmt>(LocalAddress(return_address, 0)));
    }
  }

  static bool run(IRNode *node) {
    Inliner inliner;
    bool modified = false;
    while (true) {
      node->accept(&inliner);
      if (inliner.modifier_.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }

 private:
  DelayedIRModifier modifier_;
};

const PassID InliningPass::id = "InliningPass";

namespace irpass {

bool inlining(IRNode *root,
              const CompileConfig &config,
              const InliningPass::Args &args) {
  TI_AUTO_PROF;
  return Inliner::run(root);
}

}  // namespace irpass

}  // namespace lang
}  // namespace taichi
