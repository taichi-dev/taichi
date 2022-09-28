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
      auto origin = stmt->origin->as<MatrixOfGlobalPtrStmt>();
      if (stmt->offset->is<ConstStmt>()) {
        auto offset = stmt->offset->as<ConstStmt>();
        auto lowered = std::make_unique<GlobalPtrStmt>(
            origin->snodes[offset->val.val_int()], origin->indices);
        stmt->replace_usages_with(lowered.get());
        modifier.insert_before(stmt, std::move(lowered));
        modifier.erase(stmt);
      } else {
        TI_ASSERT_INFO(origin->dynamic_indexable, "Element of the MatrixField is not dynamic indexable.\n{}", stmt->tb);
        auto stride = std::make_unique<ConstStmt>(TypedConstant(origin->dynamic_index_stride));
        auto offset = std::make_unique<BinaryOpStmt>(BinaryOpType::mul, stmt->offset, stride.get());
        auto ptr_base = std::make_unique<GlobalPtrStmt>(origin->snodes[0], origin->indices);
        auto lowered = std::make_unique<MatrixPtrStmt>(ptr_base.get(), offset.get());
        stmt->replace_usages_with(lowered.get());
        modifier.insert_before(stmt, std::move(stride));
        modifier.insert_before(stmt, std::move(offset));
        modifier.insert_before(stmt, std::move(ptr_base));
        modifier.insert_before(stmt, std::move(lowered));
        modifier.erase(stmt);
      }
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
