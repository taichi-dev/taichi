#include <variant>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {

class Scalarize : public BasicStmtVisitor {
 public:
  DelayedIRModifier modifier_;

  explicit Scalarize(IRNode *node) {
    node->accept(this);

    modifier_.modify_ir();
  }

  /*
    "val" of StoreStmt should have already been replaced by a MatrixInitStmt in
    former scalarization.

    Before:
      StoreStmt(TensorType<4 x i32>* dest, TensorType<4 x i32> val)

    After:
      addr0 = MatrixPtrStmt(TensorType<4 x i32>* dest, 0)
      addr1 = MatrixPtrStmt(TensorType<4 x i32>* dest, 1)
      addr2 = MatrixPtrStmt(TensorType<4 x i32>* dest, 2)
      addr2 = MatrixPtrStmt(TensorType<4 x i32>* dest, 3)

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

      TI_ASSERT(stmt->val->template is<MatrixInitStmt>());
      auto matrix_init_stmt = stmt->val->template as<MatrixInitStmt>();

      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = dest_tensor_type->get_element_type();
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(stmt->dest, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto scalarized_stmt = std::make_unique<T>(matrix_ptr_stmt.get(),
                                                   matrix_init_stmt->values[i]);

        modifier_.insert_before(stmt, std::move(const_stmt));
        modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }

      modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>* src)

    After:
      i32* addr0 = MatrixPtrStmt(TensorType<4 x i32>* src, 0)
      i32* addr1 = MatrixPtrStmt(TensorType<4 x i32>* src, 1)
      i32* addr2 = MatrixPtrStmt(TensorType<4 x i32>* src, 2)
      i32* addr3 = MatrixPtrStmt(TensorType<4 x i32>* src, 3)

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

      auto primitive_type = src_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(stmt->src, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto scalarized_stmt = std::make_unique<T>(matrix_ptr_stmt.get());
        scalarized_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(scalarized_stmt.get());

        modifier_.insert_before(stmt, std::move(const_stmt));
        modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
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
      auto primitive_type = operand_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto unary_stmt = std::make_unique<UnaryOpStmt>(
            stmt->op_type, operand_matrix_init_stmt->values[i]);
        unary_stmt->ret_type = primitive_type;
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

  /*
    Before:
      TensorType<4 x i32> val = BinaryStmt(TensorType<4 x i32> lhs,
                                           TensorType<4 x i32> rhs)

      * Note that "lhs" and "rhs" should have already been scalarized to
    MatrixInitStmt

    After:
      i32 calc_val0 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[0],
                                 rhs->cast<MatrixInitStmt>()->val[0])
      i32 calc_val1 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[1],
                                 rhs->cast<MatrixInitStmt>()->val[1])
      i32 calc_val2 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[2],
                                 rhs->cast<MatrixInitStmt>()->val[2])
      i32 calc_val3 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[3],
                                 rhs->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(calc_val0, calc_val1,
                           calc_val2, calc_val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(BinaryOpStmt *stmt) override {
    auto lhs_dtype = stmt->lhs->ret_type;
    auto rhs_dtype = stmt->rhs->ret_type;

    if (lhs_dtype->is<PrimitiveType>() && rhs_dtype->is<PrimitiveType>()) {
      return;
    }

    // BinaryOpExpression::type_check() should have taken care of the
    // broadcasting and neccessary conversions. So we simply add an assertion
    // here to make sure that the operands are of the same shape and dtype
    TI_ASSERT(lhs_dtype == rhs_dtype);

    if (lhs_dtype->is<TensorType>() && rhs_dtype->is<TensorType>()) {
      TI_ASSERT(lhs_dtype->cast<TensorType>()->get_num_elements() ==
                rhs_dtype->cast<TensorType>()->get_num_elements());

      auto lhs_matrix_init_stmt = stmt->lhs->cast<MatrixInitStmt>();
      std::vector<Stmt *> lhs_vals = lhs_matrix_init_stmt->values;

      auto rhs_matrix_init_stmt = stmt->rhs->cast<MatrixInitStmt>();
      std::vector<Stmt *> rhs_vals = rhs_matrix_init_stmt->values;

      TI_ASSERT(rhs_vals.size() == lhs_vals.size());

      size_t num_elements = lhs_vals.size();
      auto primitive_type = stmt->ret_type.get_element_type();
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        auto binary_stmt = std::make_unique<BinaryOpStmt>(
            stmt->op_type, lhs_vals[i], rhs_vals[i]);
        matrix_init_values.push_back(binary_stmt.get());
        binary_stmt->ret_type = primitive_type;

        modifier_.insert_before(stmt, std::move(binary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      modifier_.erase(stmt);
    }
  }

  void visit(PrintStmt *stmt) override {
    auto &contents = stmt->contents;
    std::vector<std::variant<Stmt *, std::string>> new_contents;
    for (size_t i = 0; i < contents.size(); i++) {
      auto content = contents[i];
      if (auto string_ptr = std::get_if<std::string>(&content)) {
        new_contents.push_back(*string_ptr);
      } else {
        Stmt *print_stmt = std::get<Stmt *>(content);
        if (print_stmt->is<MatrixInitStmt>()) {
          auto matrix_init_stmt = print_stmt->cast<MatrixInitStmt>();
          for (size_t j = 0; j < matrix_init_stmt->values.size(); j++) {
            new_contents.push_back(matrix_init_stmt->values[j]);
          }
        } else {
          new_contents.push_back(print_stmt);
        }
      }
    }
    modifier_.insert_before(stmt, Stmt::make<PrintStmt>(new_contents));
    modifier_.erase(stmt);
  }

  void visit(ArgLoadStmt *stmt) override {
    stmt->ret_type = stmt->ret_type.ptr_removed().get_element_type();
    stmt->ret_type.set_is_pointer(true);
  }

  /*
    Before:
      TensorType<4 x i32> val = AtomicStmt(TensorType<4 x i32>* dest,
                                           TensorType<4 x i32>  val)

    After:
      i32* dest_ptr_0 = MatrixPtrStmt(dest, 0)
      i32* dest_ptr_1 = MatrixPtrStmt(dest, 1)
      i32* dest_ptr_2 = MatrixPtrStmt(dest, 2)
      i32* dest_ptr_3 = MatrixPtrStmt(dest, 3)

      i32 dest_val0 = AtomicStmt(dest_ptr_0,
                                 val->cast<MatrixInitStmt>()->val[0])
      i32 dest_val1 = AtomicStmt(dest_ptr_1,
                                 val->cast<MatrixInitStmt>()->val[1])
      i32 dest_val2 = AtomicStmt(dest_ptr_2,
                                 val->cast<MatrixInitStmt>()->val[2])
      i32 dest_val3 = AtomicStmt(dest_ptr_3,
                                 val->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(dest_val0, dest_val1,
                           dest_val2, dest_val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(AtomicOpStmt *stmt) override {
    auto dest_dtype = stmt->dest->ret_type.ptr_removed();
    auto val_dtype = stmt->val->ret_type;

    if (dest_dtype->is<PrimitiveType>() && val_dtype->is<PrimitiveType>()) {
      return;
    }

    // AtomicOpExpression::type_check() have taken care of the broadcasting,
    // but the type conversions are delayed until irpass::type_check().
    // So we only check for the shape here.
    TI_ASSERT(dest_dtype->is<TensorType>() && val_dtype->is<TensorType>());
    TI_ASSERT(dest_dtype->cast<TensorType>()->get_shape() ==
              val_dtype->cast<TensorType>()->get_shape());

    if (dest_dtype->is<TensorType>() && val_dtype->is<TensorType>()) {
      // Scalarization for LoadStmt should have already replaced val operand
      // to MatrixInitStmt
      TI_ASSERT(stmt->val->is<MatrixInitStmt>());

      auto val_matrix_init_stmt = stmt->val->cast<MatrixInitStmt>();
      std::vector<Stmt *> val_values = val_matrix_init_stmt->values;

      size_t num_elements = val_values.size();
      auto primitive_type = stmt->ret_type.get_element_type();

      // Scalarize dest & val
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        // scalarize to dest_i
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));
        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(stmt->dest, const_stmt.get());

        // scalarize to val_i
        auto val_stmt = val_values[i];

        // assemble to scalarized atomic_op
        auto atomic_stmt = std::make_unique<AtomicOpStmt>(
            stmt->op_type, matrix_ptr_stmt.get(), val_stmt);
        atomic_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(atomic_stmt.get());

        modifier_.insert_before(stmt, std::move(const_stmt));
        modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = TernaryStmt(TensorType<4 x i32> cond,
                                            TensorType<4 x i32> lhs,
                                            TensorType<4 x i32> rhs)

    After:
      i32 val0 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[0],
                             lhs->cast<MatrixInitStmt>()->val[0],
                             rhs->cast<MatrixInitStmt>()->val[0])

      i32 val1 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[1],
                             lhs->cast<MatrixInitStmt>()->val[1],
                             rhs->cast<MatrixInitStmt>()->val[1])

      i32 val2 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[2],
                             lhs->cast<MatrixInitStmt>()->val[2],
                             rhs->cast<MatrixInitStmt>()->val[2])

      i32 val3 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[3],
                             lhs->cast<MatrixInitStmt>()->val[3],
                             rhs->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(val0, val1, val2, val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(TernaryOpStmt *stmt) override {
    auto cond_dtype = stmt->op1->ret_type;
    auto op2_dtype = stmt->op2->ret_type;
    auto op3_dtype = stmt->op3->ret_type;

    if (cond_dtype->is<PrimitiveType>() && op2_dtype->is<PrimitiveType>() &&
        op3_dtype->is<PrimitiveType>()) {
      return;
    }

    // TernaryOpExpression::type_check() have taken care of the broadcasting,
    // but the type conversions are delayed until irpass::type_check().
    // So we only check for the shape here.
    TI_ASSERT(cond_dtype.get_shape() == op2_dtype.get_shape());
    TI_ASSERT(op2_dtype.get_shape() == op3_dtype.get_shape());

    if (cond_dtype->is<TensorType>() && op2_dtype->is<TensorType>() &&
        op3_dtype->is<TensorType>()) {
      TI_ASSERT(stmt->op1->is<MatrixInitStmt>());
      TI_ASSERT(stmt->op2->is<MatrixInitStmt>());
      TI_ASSERT(stmt->op3->is<MatrixInitStmt>());

      auto cond_matrix_init_stmt = stmt->op1->cast<MatrixInitStmt>();
      std::vector<Stmt *> cond_vals = cond_matrix_init_stmt->values;

      auto op2_matrix_init_stmt = stmt->op2->cast<MatrixInitStmt>();
      std::vector<Stmt *> op2_vals = op2_matrix_init_stmt->values;

      auto op3_matrix_init_stmt = stmt->op3->cast<MatrixInitStmt>();
      std::vector<Stmt *> op3_vals = op3_matrix_init_stmt->values;

      TI_ASSERT(cond_vals.size() == op2_vals.size());
      TI_ASSERT(op2_vals.size() == op3_vals.size());

      size_t num_elements = cond_vals.size();
      auto primitive_type = stmt->ret_type.get_element_type();
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        auto ternary_stmt = std::make_unique<TernaryOpStmt>(
            stmt->op_type, cond_vals[i], op2_vals[i], op3_vals[i]);
        matrix_init_values.push_back(ternary_stmt.get());
        ternary_stmt->ret_type = primitive_type;

        modifier_.insert_before(stmt, std::move(ternary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      modifier_.erase(stmt);
    }
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

 private:
  using BasicStmtVisitor::visit;
};

class ScalarizePointers : public BasicStmtVisitor {
 public:
  DelayedIRModifier modifier_;

  // { original_alloca_stmt : [scalarized_alloca_stmt0, ...] }
  std::unordered_map<Stmt *, std::vector<Stmt *>> scalarized_local_tensor_map_;

  explicit ScalarizePointers(IRNode *node) {
    node->accept(this);

    modifier_.modify_ir();
  }

  /*
    Accessing scalar values are always more efficient than accessing elements
    from a vector - the former generates less instructions, leading to better
    performance in both compilation and runtime.

    Although we can do nothing about "global" tensors like tensors from
    ArgLoadStmt or GlobalPtrStmt, we can still optimize "local" tensors like
    tensors from AllocaStmt. In this pass, we ask AllocaStmt to allocate
    multiple scalarized PrimitiveTyped variables in replacement of the original
    TensorType.

    An additional container "scalarized_local_tensor_map_" is used to keep track
    of the scalarized AllocaStmt, for later use in LoadStmt and StoreStmt.

    Before:
      TensorType<4 x i32>* addr = AllocaStmt(TensorType<4 x i32>)

    After:
      i32 addr0 = AllocaStmt(i32)
      i32 addr1 = AllocaStmt(i32)
      i32 addr2 = AllocaStmt(i32)
      i32 addr3 = AllocaStmt(i32)

      scalarized_local_tensor_map_[addr] = {addr0, addr1, addr2, addr3}
  */
  void visit(AllocaStmt *stmt) override {
    auto tensor_type = stmt->ret_type.ptr_removed()->cast<TensorType>();
    if (tensor_type) {
      auto primitive_type = tensor_type->get_element_type();

      TI_ASSERT(scalarized_local_tensor_map_.count(stmt) == 0);
      scalarized_local_tensor_map_[stmt] = {};
      for (size_t i = 0; i < tensor_type->get_num_elements(); i++) {
        auto scalarized_alloca_stmt =
            std::make_unique<AllocaStmt>(primitive_type);
        scalarized_alloca_stmt->ret_type = primitive_type;

        scalarized_local_tensor_map_[stmt].push_back(
            scalarized_alloca_stmt.get());
        modifier_.insert_before(stmt, std::move(scalarized_alloca_stmt));
      }

      modifier_.erase(stmt);
    }
  }

  /*
    Before:
      MatrixPtrStmt(TensorType<4 x i32>* alloca_stmt, int offset)

    After:
      scalarized_alloca_stmt = scalarized_local_tensor_map_[alloca_stmt][offset]
      stmt->replace_all_usages_with(scalarized_alloca_stmt)
  */
  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AllocaStmt>()) {
      auto alloca_stmt = stmt->origin->cast<AllocaStmt>();
      auto tensor_type =
          alloca_stmt->ret_type.ptr_removed()->cast<TensorType>();
      if (tensor_type) {
        int num_elements = tensor_type->get_num_elements();
        TI_ASSERT(scalarized_local_tensor_map_.count(alloca_stmt));

        const auto &scalarized_alloca_stmts =
            scalarized_local_tensor_map_[alloca_stmt];
        TI_ASSERT(scalarized_alloca_stmts.size() == num_elements);

        // TODO(zhanlue): loose this contraint once dynamic indexing is properly
        // handled
        TI_ASSERT(stmt->offset->is<ConstStmt>());
        int offset = stmt->offset->cast<ConstStmt>()->val.val_int32();

        TI_ASSERT(offset < scalarized_alloca_stmts.size());
        auto alloca_stmt = scalarized_alloca_stmts[offset];

        stmt->replace_usages_with(alloca_stmt);
        modifier_.erase(stmt);
      }
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

namespace irpass {

void scalarize(IRNode *root) {
  TI_AUTO_PROF;
  Scalarize scalarize_pass(root);
  if (!root->get_kernel()->program->this_thread_config().dynamic_index) {
    ScalarizePointers scalarize_pointers_pass(root);
  }
}

}  // namespace irpass

}  // namespace taichi::lang
