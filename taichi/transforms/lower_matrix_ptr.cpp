#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {

class LowerMatrixPtr : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<MatrixOfGlobalPtrStmt>()) {
      TI_ASSERT(stmt->offset->is<ConstStmt>());
      auto origin = stmt->origin->as<MatrixOfGlobalPtrStmt>();
      auto offset = stmt->offset->as<ConstStmt>();
      auto lowered = std::make_unique<GlobalPtrStmt>(
          origin->snodes[offset->val.val_int()], origin->indices);
      stmt->replace_usages_with(lowered.get());
      modifier.insert_before(stmt, std::move(lowered));
      modifier.erase(stmt);
    }
  }

  static void run(IRNode *node) {
    LowerMatrixPtr pass;
    node->accept(&pass);
    pass.modifier.modify_ir();
  }
};

class RemoveMatrixOfGlobalPtr : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    modifier.erase(stmt);
  }

  static void run(IRNode *node) {
    RemoveMatrixOfGlobalPtr pass;
    node->accept(&pass);
    pass.modifier.modify_ir();
  }
};

namespace irpass {

void lower_matrix_ptr(IRNode *root) {
  TI_AUTO_PROF;
  LowerMatrixPtr::run(root);
  RemoveMatrixOfGlobalPtr::run(root);
}

}  // namespace irpass

}  // namespace taichi::lang
