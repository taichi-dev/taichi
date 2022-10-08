#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"
#include <numeric>
#include <functional>

namespace taichi::lang {

class LowerMatrixPtr : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier_;

 public:
  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<MatrixOfGlobalPtrStmt>()) {
      auto origin = stmt->origin->as<MatrixOfGlobalPtrStmt>();
      if (stmt->offset->is<ConstStmt>()) {
        auto offset = stmt->offset->as<ConstStmt>();
        auto lowered = std::make_unique<GlobalPtrStmt>(
            origin->snodes[offset->val.val_int()], origin->indices);
        stmt->replace_usages_with(lowered.get());
        modifier_.insert_before(stmt, std::move(lowered));
        modifier_.erase(stmt);
      } else {
        TI_ASSERT_INFO(
            origin->dynamic_indexable,
            "Element of the MatrixField is not dynamic indexable.\n{}",
            stmt->tb);
        auto stride = std::make_unique<ConstStmt>(
            TypedConstant(origin->dynamic_index_stride));
        auto offset = std::make_unique<BinaryOpStmt>(
            BinaryOpType::mul, stmt->offset, stride.get());
        auto ptr_base =
            std::make_unique<GlobalPtrStmt>(origin->snodes[0], origin->indices);
        auto lowered =
            std::make_unique<MatrixPtrStmt>(ptr_base.get(), offset.get());
        stmt->replace_usages_with(lowered.get());
        modifier_.insert_before(stmt, std::move(stride));
        modifier_.insert_before(stmt, std::move(offset));
        modifier_.insert_before(stmt, std::move(ptr_base));
        modifier_.insert_before(stmt, std::move(lowered));
        modifier_.erase(stmt);
      }
      return;
    }
    if (stmt->origin->is<ExternalPtrStmt>()) {
      auto origin = stmt->origin->as<ExternalPtrStmt>();
      TI_ASSERT(stmt->origin->ret_type.ptr_removed()->is<TensorType>());

      std::vector<Stmt *> indices = origin->indices;
      indices.push_back(stmt->offset);

      // MatrixPtrStmt has flattened indices, linearization of which is done
      // during IndexExpression::flatten() Here we need to modify the
      // element_dim and element_shape a little bit.
      int element_dim = -1;  // AOS Vector
      std::vector<int> element_shape = {std::accumulate(
          begin(origin->element_shape), end(origin->element_shape), 1,
          std::multiplies<int>())};

      auto fused = std::make_unique<ExternalPtrStmt>(
          origin->base_ptr, indices, element_shape, element_dim);
      fused->ret_type = stmt->ret_type;

      stmt->replace_usages_with(fused.get());
      modifier_.insert_before(stmt, std::move(fused));
      modifier_.erase(stmt);
      return;
    }
  }

  static void run(IRNode *node) {
    LowerMatrixPtr pass;
    node->accept(&pass);
    pass.modifier_.modify_ir();
  }
};

class RemoveMatrixOfGlobalPtr : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier_;

 public:
  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    modifier_.erase(stmt);
  }

  static void run(IRNode *node) {
    RemoveMatrixOfGlobalPtr pass;
    node->accept(&pass);
    pass.modifier_.modify_ir();
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
