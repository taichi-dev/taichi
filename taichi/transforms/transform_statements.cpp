#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Transform each filtered statement
class StatementsTransformer : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  StatementsTransformer(
      std::function<bool(Stmt *)> filter,
      std::function<void(Stmt *, DelayedIRModifier *)> transformer)
      : filter_(std::move(filter)), transformer_(std::move(transformer)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void maybe_transform(Stmt *stmt) {
    if (filter_(stmt)) {
      transformer_(stmt, &modifier_);
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    maybe_transform(stmt);
  }

  void visit(Stmt *stmt) override {
    maybe_transform(stmt);
  }

  static bool run(IRNode *root,
                  std::function<bool(Stmt *)> filter,
                  std::function<void(Stmt *, DelayedIRModifier *)> replacer) {
    StatementsTransformer transformer(std::move(filter), std::move(replacer));
    root->accept(&transformer);
    return transformer.modifier_.modify_ir();
  }

 private:
  std::function<bool(Stmt *)> filter_;
  std::function<void(Stmt *, DelayedIRModifier *)> transformer_;
  DelayedIRModifier modifier_;
};

namespace irpass {

bool transform_statements(
    IRNode *root,
    std::function<bool(Stmt *)> filter,
    std::function<void(Stmt *, DelayedIRModifier *)> transformer) {
  return StatementsTransformer::run(root, std::move(filter),
                                    std::move(transformer));
}

}  // namespace irpass

TLANG_NAMESPACE_END
