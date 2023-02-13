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
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  explicit Scalarize(IRNode *node) : immediate_modifier_(node) {
    node->accept(this);

    delayed_modifier_.modify_ir();
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

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }

      delayed_modifier_.erase(stmt);
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

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = src_dtype;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
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
      auto primitive_type = stmt->ret_type.get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto unary_stmt = std::make_unique<UnaryOpStmt>(
            stmt->op_type, operand_matrix_init_stmt->values[i]);
        if (stmt->is_cast()) {
          unary_stmt->cast_type = stmt->cast_type.get_element_type();
        }
        unary_stmt->ret_type = primitive_type;
        matrix_init_values.push_back(unary_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(unary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = operand_dtype;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
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
    if (lhs_dtype->is<TensorType>() || rhs_dtype->is<TensorType>()) {
      // Make sure broadcasting has been correctly applied by
      // BinaryOpExpression::type_check().
      TI_ASSERT(lhs_dtype->is<TensorType>() && rhs_dtype->is<TensorType>());
      // However, since the type conversions are delayed until
      // irpass::type_check(), we only check for the shape here.
      TI_ASSERT(lhs_dtype->cast<TensorType>()->get_shape() ==
                rhs_dtype->cast<TensorType>()->get_shape());
      // Scalarization for LoadStmt should have already replaced both operands
      // to MatrixInitStmt.
      TI_ASSERT(stmt->lhs->is<MatrixInitStmt>());
      TI_ASSERT(stmt->rhs->is<MatrixInitStmt>());

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

        delayed_modifier_.insert_before(stmt, std::move(binary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
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
          auto tensor_shape =
              print_stmt->ret_type->as<TensorType>()->get_shape();

          bool is_matrix = tensor_shape.size() == 2;
          int m = tensor_shape[0];

          new_contents.push_back("[");
          if (is_matrix) {
            int n = tensor_shape[1];
            for (size_t i = 0; i < m; i++) {
              new_contents.push_back("[");
              for (size_t j = 0; j < n; j++) {
                size_t index = i * n + j;
                new_contents.push_back(matrix_init_stmt->values[index]);
                if (j != n - 1)
                  new_contents.push_back(", ");
              }
              new_contents.push_back("]");

              if (i != m - 1)
                new_contents.push_back(", ");
            }
          } else {
            for (size_t i = 0; i < m; i++) {
              new_contents.push_back(matrix_init_stmt->values[i]);
              if (i != m - 1)
                new_contents.push_back(", ");
            }
          }
          new_contents.push_back("]");
        } else {
          new_contents.push_back(print_stmt);
        }
      }
    }

    // Merge string contents
    std::vector<std::variant<Stmt *, std::string>> merged_contents;
    std::string merged_string = "";
    for (const auto &content : new_contents) {
      if (auto string_content = std::get_if<std::string>(&content)) {
        merged_string += *string_content;
      } else {
        if (!merged_string.empty()) {
          merged_contents.push_back(merged_string);
          merged_string = "";
        }
        merged_contents.push_back(content);
      }
    }
    if (!merged_string.empty())
      merged_contents.push_back(merged_string);

    delayed_modifier_.insert_before(stmt,
                                    Stmt::make<PrintStmt>(merged_contents));
    delayed_modifier_.erase(stmt);
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
    if (dest_dtype->is<TensorType>() || val_dtype->is<TensorType>()) {
      // Make sure broadcasting has been correctly applied by
      // AtomicOpExpression::type_check().
      TI_ASSERT(dest_dtype->is<TensorType>() && val_dtype->is<TensorType>());
      // However, since the type conversions are delayed until
      // irpass::type_check(), we only check for the shape here.
      TI_ASSERT(dest_dtype->cast<TensorType>()->get_shape() ==
                val_dtype->cast<TensorType>()->get_shape());
      // Scalarization for LoadStmt should have already replaced val operand
      // to MatrixInitStmt.
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

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
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
    if (cond_dtype->is<TensorType>()) {
      // Make sure broadcasting has been correctly applied by
      // TernaryOpExpression::type_check().
      TI_ASSERT(cond_dtype->is<TensorType>() && op2_dtype->is<TensorType>() &&
                op3_dtype->is<TensorType>());
      // However, since the type conversions are delayed until
      // irpass::type_check(), we only check for the shape here.
      TI_ASSERT(cond_dtype.get_shape() == op2_dtype.get_shape());
      TI_ASSERT(op2_dtype.get_shape() == op3_dtype.get_shape());
      // Scalarization for LoadStmt should have already replaced all operands
      // to MatrixInitStmt.
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

        delayed_modifier_.insert_before(stmt, std::move(ternary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
    } else if (cond_dtype->is<PrimitiveType>() &&
               (op2_dtype->is<TensorType>() || op3_dtype->is<TensorType>())) {
      TI_ASSERT(cond_dtype->is<PrimitiveType>() &&
                op2_dtype->is<TensorType>() && op3_dtype->is<TensorType>());
      TI_ASSERT(op2_dtype.get_shape() == op3_dtype.get_shape());
      // Scalarization for LoadStmt should have already replaced all operands
      // to MatrixInitStmt.
      TI_ASSERT(stmt->op2->is<MatrixInitStmt>());
      TI_ASSERT(stmt->op3->is<MatrixInitStmt>());

      Stmt* cond_val = stmt->op1;

      auto op2_matrix_init_stmt = stmt->op2->cast<MatrixInitStmt>();
      std::vector<Stmt *> op2_vals = op2_matrix_init_stmt->values;

      auto op3_matrix_init_stmt = stmt->op3->cast<MatrixInitStmt>();
      std::vector<Stmt *> op3_vals = op3_matrix_init_stmt->values;

      TI_ASSERT(op2_vals.size() == op3_vals.size());

      size_t num_elements = op2_vals.size();
      auto primitive_type = stmt->ret_type.get_element_type();
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        auto ternary_stmt = std::make_unique<TernaryOpStmt>(
            stmt->op_type, cond_val, op2_vals[i], op3_vals[i]);
        matrix_init_values.push_back(ternary_stmt.get());
        ternary_stmt->ret_type = primitive_type;

        delayed_modifier_.insert_before(stmt, std::move(ternary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
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

  void visit(ArgLoadStmt *stmt) override {
    auto ret_type = stmt->ret_type.ptr_removed().get_element_type();
    auto arg_load = std::make_unique<ArgLoadStmt>(stmt->arg_id, ret_type,
                                                  stmt->is_ptr, stmt->is_grad);

    immediate_modifier_.replace_usages_with(stmt, arg_load.get());

    delayed_modifier_.insert_before(stmt, std::move(arg_load));
    delayed_modifier_.erase(stmt);
  }

 private:
  using BasicStmtVisitor::visit;
};

// The GatherScalarizableLocalPointers gathers all local TensorType allocas
// only indexed with constants, which can then be scalarized in the
// ScalarizeLocalPointers pass.
class GatherScalarizableLocalPointers : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;

  std::unordered_map<Stmt *, bool> is_alloca_scalarizable_;

 public:
  void visit(AllocaStmt *stmt) override {
    if (stmt->ret_type.ptr_removed()->is<TensorType>()) {
      TI_ASSERT(is_alloca_scalarizable_.count(stmt) == 0);
      is_alloca_scalarizable_[stmt] = !stmt->is_shared;
    }
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AllocaStmt>()) {
      TI_ASSERT(is_alloca_scalarizable_.count(stmt->origin) == 1);
      if (!stmt->offset->is<ConstStmt>()) {
        is_alloca_scalarizable_[stmt->origin] = false;
      }
    }
  }

  static std::unordered_set<Stmt *> run(IRNode *node) {
    GatherScalarizableLocalPointers pass;
    node->accept(&pass);
    std::unordered_set<Stmt *> result;
    for (auto &[k, v] : pass.is_alloca_scalarizable_) {
      if (v) {
        result.insert(k);
      }
    }
    return result;
  }
};

class ScalarizeLocalPointers : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  std::unordered_set<Stmt *> scalarizable_allocas_;
  // { original_alloca_stmt : [scalarized_alloca_stmt0, ...] }
  std::unordered_map<Stmt *, std::vector<Stmt *>> scalarized_local_tensor_map_;

  explicit ScalarizeLocalPointers(
      IRNode *node,
      const std::unordered_set<Stmt *> &scalarizable_allocas)
      : immediate_modifier_(node), scalarizable_allocas_(scalarizable_allocas) {
    node->accept(this);

    delayed_modifier_.modify_ir();
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
    if (scalarizable_allocas_.count(stmt) == 1) {
      auto tensor_type = stmt->ret_type.ptr_removed()->cast<TensorType>();
      TI_ASSERT(tensor_type != nullptr);
      auto primitive_type = tensor_type->get_element_type();

      TI_ASSERT(scalarized_local_tensor_map_.count(stmt) == 0);
      scalarized_local_tensor_map_[stmt] = {};
      for (size_t i = 0; i < tensor_type->get_num_elements(); i++) {
        auto scalarized_alloca_stmt =
            std::make_unique<AllocaStmt>(primitive_type);
        scalarized_alloca_stmt->ret_type = primitive_type;

        scalarized_local_tensor_map_[stmt].push_back(
            scalarized_alloca_stmt.get());
        delayed_modifier_.insert_before(stmt,
                                        std::move(scalarized_alloca_stmt));
      }

      delayed_modifier_.erase(stmt);
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
    if (stmt->origin->is<AllocaStmt>() &&
        scalarizable_allocas_.count(stmt->origin) == 1) {
      auto alloca_stmt = stmt->origin->cast<AllocaStmt>();
      auto tensor_type =
          alloca_stmt->ret_type.ptr_removed()->cast<TensorType>();
      TI_ASSERT(tensor_type != nullptr);
      int num_elements = tensor_type->get_num_elements();
      TI_ASSERT(scalarized_local_tensor_map_.count(alloca_stmt));

      const auto &scalarized_alloca_stmts =
          scalarized_local_tensor_map_[alloca_stmt];
      TI_ASSERT(scalarized_alloca_stmts.size() == num_elements);

      TI_ASSERT(stmt->offset->is<ConstStmt>());
      int offset = stmt->offset->cast<ConstStmt>()->val.val_int32();

      TI_ASSERT(offset < scalarized_alloca_stmts.size());
      auto new_stmt = scalarized_alloca_stmts[offset];

      immediate_modifier_.replace_usages_with(stmt, new_stmt);
      delayed_modifier_.erase(stmt);
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

// The ExtractLocalPointers pass aims at removing redundant ConstStmts and
// MatrixPtrStmts generated for any (AllocaStmt, integer) pair by extracting
// a unique copy for any future usage.
//
// Example for redundant stmts:
//   <i32> $0 = const 0
//   <i32> $1 = const 1
//   ...
//   <[Tensor (3, 3) f32]> $47738 = alloca
//   <i32> $47739 = const 0  [REDUNDANT]
//   <*f32> $47740 = shift ptr [$47738 + $47739]
//   $47741 : local store [$47740 <- $47713]
//   <i32> $47742 = const 1  [REDUNDANT]
//   <*f32> $47743 = shift ptr [$47738 + $47742]
//   $47744 : local store [$47743 <- $47716]
//   ...
//   <i32> $47812 = const 1  [REDUNDANT]
//   <*f32> $47813 = shift ptr [$47738 + $47812]  [REDUNDANT]
//   <f32> $47814 = local load [$47813]
class ExtractLocalPointers : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  std::unordered_map<std::pair<Stmt *, int>,
                     Stmt *,
                     hashing::Hasher<std::pair<Stmt *, int>>>
      first_matrix_ptr_;  // mapping an (AllocaStmt, integer) pair to the first
                          // MatrixPtrStmt representing it
  std::unordered_map<int, Stmt *>
      first_const_;  // mapping an integer to the first ConstStmt representing
                     // it
  Block *top_level_;

  explicit ExtractLocalPointers(IRNode *root) : immediate_modifier_(root) {
    TI_ASSERT(root->is<Block>());
    top_level_ = root->as<Block>();
    root->accept(this);
    delayed_modifier_.modify_ir();
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AllocaStmt>()) {
      auto alloca_stmt = stmt->origin->cast<AllocaStmt>();
      auto tensor_type =
          alloca_stmt->ret_type.ptr_removed()->cast<TensorType>();
      TI_ASSERT(tensor_type != nullptr);
      if (stmt->offset->is<ConstStmt>()) {
        int offset = stmt->offset->cast<ConstStmt>()->val.val_int32();
        if (first_const_.count(offset) == 0) {
          first_const_[offset] = stmt->offset;
          delayed_modifier_.extract_to_block_front(stmt->offset, top_level_);
        }
        auto key = std::make_pair(alloca_stmt, offset);
        if (first_matrix_ptr_.count(key) == 0) {
          auto extracted = std::make_unique<MatrixPtrStmt>(
              alloca_stmt, first_const_[offset]);
          first_matrix_ptr_[key] = extracted.get();
          delayed_modifier_.insert_after(alloca_stmt, std::move(extracted));
        }
        auto new_stmt = first_matrix_ptr_[key];
        immediate_modifier_.replace_usages_with(stmt, new_stmt);
        delayed_modifier_.erase(stmt);
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
  auto scalarizable_allocas = GatherScalarizableLocalPointers::run(root);
  ScalarizeLocalPointers scalarize_pointers_pass(root, scalarizable_allocas);
  ExtractLocalPointers extract_pointers_pass(root);
}

}  // namespace irpass

}  // namespace taichi::lang
