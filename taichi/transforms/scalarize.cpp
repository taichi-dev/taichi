#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

class Scalarize : public IRVisitor {
 public:
  DelayedIRModifier modifier_;

  Scalarize(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
    node->accept(this);

    modifier_.modify_ir();
  }

  /*
    "val" of StoreStmt should have already been replaced by a MatrixInitStmt in
    former scalarization.

    Before:
      StoreStmt(TensorType<4 x i32>* dest, TensorType<4 x i32> val)

    After:
      addr0 = PtrOffsetStmt(TensorType<4 x i32>* dest, 0)
      addr1 = PtrOffsetStmt(TensorType<4 x i32>* dest, 1)
      addr2 = PtrOffsetStmt(TensorType<4 x i32>* dest, 2)
      addr2 = PtrOffsetStmt(TensorType<4 x i32>* dest, 3)

      StoreStmt(i32* addr0, i32 val->cast<MatrixInitStmt>()->val[0])
      StoreStmt(i32* addr1, i32 val->cast<MatrixInitStmt>()->val[1])
      StoreStmt(i32* addr2, i32 val->cast<MatrixInitStmt>()->val[2])
      StoreStmt(i32* addr3, i32 val->cast<MatrixInitStmt>()->val[3])
  */
  template <typename T>
  void scalarize_store_stmt(T *stmt) {
    auto dest_dtype = stmt->dest->ret_type.ptr_removed();
    auto val_dtype = stmt->val->ret_type;
    if (dest_dtype->template is<TensorType>() &&
        val_dtype->template is<TensorType>()) {
      // Needs scalarize
      auto dest_tensor_type = dest_dtype->template as<TensorType>();
      auto val_tensor_type = val_dtype->template as<TensorType>();

      TI_ASSERT(dest_tensor_type->get_shape() == val_tensor_type->get_shape());
      // For sqrt/exp/log with int-type operand, we automatically set the
      // ret_type to float32. In that case the dtype of dest and val may be
      // different, and we rely on the following type_promotion() and
      // load_store_forwarding() to handle this situation.

      TI_ASSERT(stmt->val->template is<MatrixInitStmt>());
      auto matrix_init_stmt = stmt->val->template as<MatrixInitStmt>();

      int num_elements = val_tensor_type->get_num_elements();
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        auto ptr_offset_stmt =
            std::make_unique<PtrOffsetStmt>(stmt->dest, const_stmt.get());
        auto scalarized_stmt = std::make_unique<T>(ptr_offset_stmt.get(),
                                                   matrix_init_stmt->values[i]);

        modifier_.insert_before(stmt, std::move(const_stmt));
        modifier_.insert_before(stmt, std::move(ptr_offset_stmt));
        modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }
      modifier_.erase(stmt);
    }
  }

  /*

    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>* src)

    After:
      i32* addr0 = PtrOffsetStmt(TensorType<4 x i32>* src, 0)
      i32* addr1 = PtrOffsetStmt(TensorType<4 x i32>* src, 1)
      i32* addr2 = PtrOffsetStmt(TensorType<4 x i32>* src, 2)
      i32* addr3 = PtrOffsetStmt(TensorType<4 x i32>* src, 3)

      i32 val0 = LoadStmt(addr0)
      i32 val1 = LoadStmt(addr1)
      i32 val2 = LoadStmt(addr2)
      i32 val3 = LoadStmt(addr3)

      tmp = MatrixInitStmt(val0, val1, val2, val3)

      stmt->replace_all_usages_with(tmp)
  */
  template <typename T>
  void scalarize_load_stmt(T *stmt) {
    auto src_dtype = stmt->src->ret_type.ptr_removed();
    if (src_dtype->template is<TensorType>()) {
      // Needs scalarize
      auto src_tensor_type = src_dtype->template as<TensorType>();

      std::vector<Stmt *> matrix_init_values;
      int num_elements = src_tensor_type->get_num_elements();

      for (size_t i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        auto ptr_offset_stmt =
            std::make_unique<PtrOffsetStmt>(stmt->src, const_stmt.get());
        auto scalarized_stmt = std::make_unique<T>(ptr_offset_stmt.get());

        matrix_init_values.push_back(scalarized_stmt.get());

        modifier_.insert_before(stmt, std::move(const_stmt));
        modifier_.insert_before(stmt, std::move(ptr_offset_stmt));
        modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);

      matrix_init_stmt->ret_type = src_dtype;

      stmt->replace_usages_with(matrix_init_stmt.get());
      modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      modifier_.erase(stmt);
    }
  }

  /*

    Before:
      TensorType<4 x i32> val = UnaryStmt(TensorType<4 x i32> operand)

      * Note that "operand" should have already been scalarized to
    MatrixInitStmt

    After:
      i32 calc_val0 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[0])
      i32 calc_val1 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[1])
      i32 calc_val2 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[2])
      i32 calc_val3 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(calc_val0, calc_val1,
                           calc_val2, calc_val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(UnaryOpStmt *stmt) override {
    auto operand_dtype = stmt->operand->ret_type;
    if (operand_dtype->is<TensorType>()) {
      // Needs scalarize
      auto operand_tensor_type = operand_dtype->as<TensorType>();

      TI_ASSERT(stmt->operand->is<MatrixInitStmt>());
      auto operand_matrix_init_stmt = stmt->operand->cast<MatrixInitStmt>();

      TI_ASSERT(operand_matrix_init_stmt->values.size() ==
                operand_tensor_type->get_num_elements());

      std::vector<Stmt *> matrix_init_values;
      int num_elements = operand_tensor_type->get_num_elements();
      for (size_t i = 0; i < num_elements; i++) {
        auto unary_stmt = std::make_unique<UnaryOpStmt>(
            stmt->op_type, operand_matrix_init_stmt->values[i]);
        matrix_init_values.push_back(unary_stmt.get());

        modifier_.insert_before(stmt, std::move(unary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = operand_dtype;

      stmt->replace_usages_with(matrix_init_stmt.get());
      modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      modifier_.erase(stmt);
    }
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(MeshForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    stmt->all_blocks_accept(this);
  }

  void visit(GlobalStoreStmt *stmt) override {
    scalarize_store_stmt<GlobalStoreStmt>(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    scalarize_store_stmt<LocalStoreStmt>(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    scalarize_load_stmt<GlobalLoadStmt>(stmt);
  }

  void visit(LocalLoadStmt *stmt) override {
    scalarize_load_stmt<LocalLoadStmt>(stmt);
  }
};

namespace irpass {

void scalarize(IRNode *root) {
  TI_AUTO_PROF;

  Scalarize scalarize_pass(root);

  /* TODO(zhanlue): Remove redundant MatrixInitStmt
    Scalarize pass will generate temporary MatrixInitStmts, which are only used
    as rvalues. Remove these MatrixInitStmts since it's no longer needed.
  */
}

}  // namespace irpass

TLANG_NAMESPACE_END
