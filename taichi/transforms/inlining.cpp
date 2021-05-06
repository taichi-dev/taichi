#include "taichi/transforms/inlining.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

namespace taichi {
namespace lang {

// Inline all functions.
class Inlining : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;
  Program *program;

  explicit Inlining(Program *program) : BasicStmtVisitor(), program(program) {
  }

  void visit(FuncCallStmt *stmt) override {
    auto *func = program->function_map[stmt->funcid];
    TI_ASSERT(func);
    TI_ASSERT(func->args.size() == stmt->args.size());
    TI_ASSERT(func->ir->is<Block>());
    TI_ASSERT(func->rets.size() <= 1);
    auto inlined_ir = irpass::analysis::clone(func->ir.get());
    if (!func->args.empty()) {
      // TODO: Make sure that if stmt->args is an ArgLoadStmt,
      //  it will not be replaced again here
      irpass::replace_statements_with(
          inlined_ir.get(),
          /*filter=*/[&](Stmt *s) { return s->is<ArgLoadStmt>(); },
          /*generator=*/
          [&](Stmt *s) { return stmt->args[s->as<ArgLoadStmt>()->arg_id]; });
    }
    if (!func->rets.empty()) {
      if (irpass::analysis::gather_statements(
              inlined_ir.get(),
              [&](Stmt *s) { return s->is<KernelReturnStmt>(); })
              .size() > 1) {
        TI_WARN(
            "Multiple returns in function \"{}\" may not be handled properly.",
            func->funcid);
      }
      // Use a local variable to store the return value
      auto *return_address = inlined_ir->as<Block>()->insert(
          Stmt::make<AllocaStmt>(func->rets[0].dt), /*location=*/0);
      irpass::replace_statements_with(
          inlined_ir.get(),
          /*filter=*/[&](Stmt *s) { return s->is<KernelReturnStmt>(); },
          /*generator=*/
          [&](Stmt *s) {
            return Stmt::make<LocalStoreStmt>(return_address,
                                              s->as<KernelReturnStmt>()->value);
          });
      modifier.insert_before(stmt,
                             std::move(inlined_ir->as<Block>()->statements));
      // Load the return value here
      modifier.replace_with(
          stmt, Stmt::make<LocalLoadStmt>(LocalAddress(return_address, 0)));
    } else {
      modifier.replace_with(stmt,
                            std::move(inlined_ir->as<Block>()->statements));
    }
  }

  static bool run(IRNode *node, Program *program) {
    Inlining inliner(program);
    bool modified = false;
    while (true) {
      node->accept(&inliner);
      if (inliner.modifier.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }
};

const PassID InliningPass::id = "InliningPass";

namespace irpass {

bool inlining(IRNode *root,
              const CompileConfig &config,
              const InliningPass::Args &args) {
  TI_AUTO_PROF;
  return Inlining::run(root, args.program);
}

}  // namespace irpass

}  // namespace lang
}  // namespace taichi
