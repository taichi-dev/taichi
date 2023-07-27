#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"
#include <numeric>
#include <functional>

namespace taichi::lang {

bool is_aos_matrix_of_global_ptr(MatrixOfGlobalPtrStmt *stmt) {
  auto ret_type = stmt->ret_type;
  auto indices = stmt->indices;
  auto snodes = stmt->snodes;

  bool is_continuous_addr = true;
  SNode *parent_snode = snodes[0]->parent;
  for (auto snode : snodes) {
    TI_ASSERT(snode->type == SNodeType::place);
    if (snode->parent != parent_snode ||
        parent_snode->type != SNodeType::dense) {
      is_continuous_addr = false;
      break;
    }
  }

  return is_continuous_addr;
}

class GatherValidAOSGlobalPtrStmt : public BasicStmtVisitor {
 public:
  DelayedIRModifier modifier_;
  std::unordered_set<Stmt *> invalid_aos_global_ptr_stmts_;

  explicit GatherValidAOSGlobalPtrStmt(IRNode *node) {
    node->accept(this);

    modifier_.modify_ir();
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<MatrixOfGlobalPtrStmt>()) {
      auto origin = stmt->origin->as<MatrixOfGlobalPtrStmt>();

      if (is_quant(origin->snodes[0]->dt)) {
        invalid_aos_global_ptr_stmts_.insert(stmt->origin);
      }

      if (origin->snodes[0]->dt->is<PointerType>() &&
          origin->snodes[0]->dt->as<PointerType>()->is_bit_pointer()) {
        invalid_aos_global_ptr_stmts_.insert(stmt->origin);
      }

      if (!stmt->offset->is<ConstStmt>()) {
        invalid_aos_global_ptr_stmts_.insert(stmt->origin);
      }
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

class LowerAOSGlobalPtrStmt : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;
  std::unordered_set<Stmt *> invalid_aos_global_ptr_stmts_;

  explicit LowerAOSGlobalPtrStmt(
      IRNode *node,
      std::unordered_set<Stmt *> &invalid_aos_global_ptr_stmts)
      : immediate_modifier_(node),
        invalid_aos_global_ptr_stmts_(invalid_aos_global_ptr_stmts) {
    node->accept(this);

    delayed_modifier_.modify_ir();
  }

  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    bool is_aos = is_aos_matrix_of_global_ptr(stmt);
    auto ret_type = stmt->ret_type;
    auto indices = stmt->indices;
    auto snodes = stmt->snodes;

    if (is_aos && invalid_aos_global_ptr_stmts_.find(stmt) ==
                      invalid_aos_global_ptr_stmts_.end()) {
      auto new_stmt = std::make_unique<GlobalPtrStmt>(snodes[0], indices);
      new_stmt->ret_type = ret_type;
      new_stmt->ret_type.set_is_pointer(true);

      stmt->replace_usages_with(new_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(new_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

class ScalarizeMatrixPtr : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  explicit ScalarizeMatrixPtr(IRNode *node) : immediate_modifier_(node) {
    node->accept(this);

    delayed_modifier_.modify_ir();
  }

  /*
    [MatrixOfGlobalPtrStmt]
    Before:
      StoreStmt(TensorType<4 x i32>* MatrixOfGlobalPtrStmt, TensorType<4 x i32>
    val)

    After:
      AllocaStmt alloca = AllocaStmt(TensorType<4 x i32>)
      StoreStmt(alloca, val)

      matrix_ptr0 = MatrixPtrStmt(i32* alloca, 0)
      matrix_ptr1 = MatrixPtrStmt(i32* alloca, 1)
      matrix_ptr2 = MatrixPtrStmt(i32* alloca, 2)
      matrix_ptr3 = MatrixPtrStmt(i32* alloca, 3)

      val0 = LoadStmt(matrix_ptr0)
      val1 = LoadStmt(matrix_ptr1)
      val2 = LoadStmt(matrix_ptr2)
      val3 = LoadStmt(matrix_ptr3)

      GlobalPtrStmt* ptr0 = MatrixOfGlobalPtrStmt->snodes[0]
      GlobalPtrStmt* ptr1 = MatrixOfGlobalPtrStmt->snodes[1]
      GlobalPtrStmt* ptr2 = MatrixOfGlobalPtrStmt->snodes[2]
      GlobalPtrStmt* ptr3 = MatrixOfGlobalPtrStmt->snodes[3]
      StoreStmt(i32* ptr0, i32 val0)
      StoreStmt(i32* ptr1, i32 val1)
      StoreStmt(i32* ptr2, i32 val2)
      StoreStmt(i32* ptr3, i32 val3)

    [MatrixOfMatrixPtrStmt]
    Before:
      StoreStmt(TensorType<4 x i32>* MatrixOfMatrixPtrStmt, TensorType<4 x i32>
    val)

    After:
      stmt0 = MatrixOfMatrixPtrStmt->stmts[0]
      stmt1 = MatrixOfMatrixPtrStmt->stmts[1]
      stmt2 = MatrixOfMatrixPtrStmt->stmts[2]
      stmt3 = MatrixOfMatrixPtrStmt->stmts[3]

      AllocaStmt alloca = AllocaStmt(TensorType<4 x i32>)
      LocalStoreStmt(alloca, val)

      matrix_ptr0 = MatrixPtrStmt(i32* alloca, 0)
      matrix_ptr1 = MatrixPtrStmt(i32* alloca, 1)
      matrix_ptr2 = MatrixPtrStmt(i32* alloca, 2)
      matrix_ptr3 = MatrixPtrStmt(i32* alloca, 3)

      val0 = LoadStmt(matrix_ptr0)
      val1 = LoadStmt(matrix_ptr1)
      val2 = LoadStmt(matrix_ptr2)
      val3 = LoadStmt(matrix_ptr3)

      StoreStmt(i32* stmt0, i32 val0)
      StoreStmt(i32* stmt1, i32 val1)
      StoreStmt(i32* stmt2, i32 val2)
      StoreStmt(i32* stmt3, i32 val3)
  */
  template <typename T>
  void scalarize_store_stmt(T *stmt) {
    if (stmt->dest->template is<MatrixOfGlobalPtrStmt>()) {
      auto matrix_of_global_ptr_stmt =
          stmt->dest->template as<MatrixOfGlobalPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->template as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto global_ptr_stmt = std::make_unique<GlobalPtrStmt>(
            matrix_of_global_ptr_stmt->snodes[i],
            matrix_of_global_ptr_stmt->indices);
        global_ptr_stmt->ret_type.set_is_pointer(true);

        auto store_stmt =
            std::make_unique<T>(global_ptr_stmt.get(), load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(global_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      }

      delayed_modifier_.erase(stmt);

    } else if (stmt->dest->template is<MatrixOfMatrixPtrStmt>()) {
      auto matrix_of_matrix_ptr_stmt =
          stmt->dest->template as<MatrixOfMatrixPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->template as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto store_stmt = std::make_unique<T>(
            matrix_of_matrix_ptr_stmt->stmts[i], load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      }

      delayed_modifier_.erase(stmt);
    }
  }

  /*
    [MatrixOfGlobalPtrStmt]
    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>*
    MatrixOfGlobalPtrStmt)

    After:
      GlobalPtrStmt* ptr0 = MatrixOfGlobalPtrStmt->snodes[0]
      GlobalPtrStmt* ptr1 = MatrixOfGlobalPtrStmt->snodes[1]
      GlobalPtrStmt* ptr2 = MatrixOfGlobalPtrStmt->snodes[2]
      GlobalPtrStmt* ptr3 = MatrixOfGlobalPtrStmt->snodes[3]

      i32 val0 = LoadStmt(ptr0)
      i32 val1 = LoadStmt(ptr1)
      i32 val2 = LoadStmt(ptr2)
      i32 val3 = LoadStmt(ptr3)

      tmp = MatrixInitStmt(val0, val1, val2, val3)
      stmt->replace_all_usages_with(tmp)

    [MatrixOfMatrixPtrStmt]
    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>*
    MatrixOfMatrixPtrStmt)

    After:
      i32* ptr0 = MatrixOfMatrixPtr->stmts[0] // usually it's a MatrixPtrStmt
      i32* ptr1 = MatrixOfMatrixPtr->stmts[1] // usually it's a MatrixPtrStmt
      i32* ptr2 = MatrixOfMatrixPtr->stmts[2] // usually it's a MatrixPtrStmt
      i32* ptr3 = MatrixOfMatrixPtr->stmts[3] // usually it's a MatrixPtrStmt

      i32 val0 = LoadStmt(ptr0)
      i32 val1 = LoadStmt(ptr1)
      i32 val2 = LoadStmt(ptr2)
      i32 val3 = LoadStmt(ptr3)

      tmp = MatrixInitStmt(val0, val1, val2, val3)
      stmt->replace_all_usages_with(tmp)
  */
  template <typename T>
  void scalarize_load_stmt(T *stmt) {
    if (stmt->src->template is<MatrixOfGlobalPtrStmt>()) {
      auto matrix_of_global_ptr_stmt =
          stmt->src->template as<MatrixOfGlobalPtrStmt>();
      auto dest_tensor_type = stmt->ret_type->template as<TensorType>();

      std::vector<Stmt *> matrix_init_values;
      int num_elements = dest_tensor_type->get_num_elements();

      auto primitive_type = dest_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto global_ptr_stmt = std::make_unique<GlobalPtrStmt>(
            matrix_of_global_ptr_stmt->snodes[i],
            matrix_of_global_ptr_stmt->indices);
        global_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<T>(global_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(global_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);

    } else if (stmt->src->template is<MatrixOfMatrixPtrStmt>()) {
      auto matrix_of_matrix_ptr_stmt =
          stmt->src->template as<MatrixOfMatrixPtrStmt>();
      auto dest_tensor_type = stmt->ret_type->template as<TensorType>();

      std::vector<Stmt *> matrix_init_values;
      int num_elements = dest_tensor_type->get_num_elements();

      auto primitive_type = dest_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        Stmt *matrix_ptr_stmt = matrix_of_matrix_ptr_stmt->stmts[i];

        auto load_stmt = std::make_unique<T>(matrix_ptr_stmt);
        load_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
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

  /*
    [MatrixOfGlobalPtrStmt]
    Before:
      StoreStmt(TensorType<4 x i32>* MatrixOfGlobalPtrStmt, TensorType<4 x i32>
    val)

    After:
      AllocaStmt alloca = AllocaStmt(TensorType<4 x i32>)
      StoreStmt(alloca, val)

      matrix_ptr0 = MatrixPtrStmt(i32* alloca, 0)
      matrix_ptr1 = MatrixPtrStmt(i32* alloca, 1)
      matrix_ptr2 = MatrixPtrStmt(i32* alloca, 2)
      matrix_ptr3 = MatrixPtrStmt(i32* alloca, 3)

      val0 = LoadStmt(matrix_ptr0)
      val1 = LoadStmt(matrix_ptr1)
      val2 = LoadStmt(matrix_ptr2)
      val3 = LoadStmt(matrix_ptr3)

      GlobalPtrStmt* ptr0 = MatrixOfGlobalPtrStmt->snodes[0]
      GlobalPtrStmt* ptr1 = MatrixOfGlobalPtrStmt->snodes[1]
      GlobalPtrStmt* ptr2 = MatrixOfGlobalPtrStmt->snodes[2]
      GlobalPtrStmt* ptr3 = MatrixOfGlobalPtrStmt->snodes[3]
      i32 val_o0 = AtomicStmt(i32* ptr0, i32 val0)
      i32 val_o1 = AtomicStmt(i32* ptr1, i32 val1)
      i32 val_o2 = AtomicStmt(i32* ptr2, i32 val2)
      i32 val_o3 = AtomicStmt(i32* ptr3, i32 val3)

      tmp = MatrixInitStmt(val_o0, val_o1, val_o2, val_o3)
      stmt->replace_all_usages_with(tmp)
  */
  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest->is<MatrixOfGlobalPtrStmt>()) {
      auto matrix_of_global_ptr_stmt = stmt->dest->as<MatrixOfGlobalPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      std::vector<Stmt *> matrix_init_values;

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto global_ptr_stmt = std::make_unique<GlobalPtrStmt>(
            matrix_of_global_ptr_stmt->snodes[i],
            matrix_of_global_ptr_stmt->indices);
        global_ptr_stmt->ret_type.set_is_pointer(true);

        auto atomic_stmt = std::make_unique<AtomicOpStmt>(
            stmt->op_type, global_ptr_stmt.get(), load_stmt.get());
        atomic_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(atomic_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(global_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);

    } else if (stmt->dest->is<MatrixOfMatrixPtrStmt>()) {
      auto matrix_of_matrix_ptr_stmt = stmt->dest->as<MatrixOfMatrixPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      std::vector<Stmt *> matrix_init_values;

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto atomic_stmt = std::make_unique<AtomicOpStmt>(
            stmt->op_type, matrix_of_matrix_ptr_stmt->stmts[i],
            load_stmt.get());
        atomic_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(atomic_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

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
        lowered->ret_type.set_is_pointer(true);
        stmt->replace_usages_with(lowered.get());
        modifier_.insert_before(stmt, std::move(lowered));
        modifier_.erase(stmt);
      } else {
        TI_ASSERT_INFO(
            origin->dynamic_indexable,
            "Element of the MatrixField is not dynamic indexable.\n{}",
            stmt->get_tb());
        auto stride = std::make_unique<ConstStmt>(
            TypedConstant(origin->dynamic_index_stride));
        auto offset = std::make_unique<BinaryOpStmt>(
            BinaryOpType::mul, stmt->offset, stride.get());
        offset->ret_type = stmt->offset->ret_type;

        auto ptr_base =
            std::make_unique<GlobalPtrStmt>(origin->snodes[0], origin->indices);
        ptr_base->ret_type.set_is_pointer(true);
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
    if (stmt->origin->is<MatrixOfMatrixPtrStmt>()) {
      auto origin = stmt->origin->as<MatrixOfMatrixPtrStmt>();
      TI_ASSERT(stmt->offset->is<ConstStmt>());
      auto offset = stmt->offset->as<ConstStmt>();
      stmt->replace_usages_with(origin->stmts[offset->val.val_int()]);
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

class RemoveMatrixOfPtr : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier_;

 public:
  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    modifier_.erase(stmt);
  }

  void visit(MatrixOfMatrixPtrStmt *stmt) override {
    modifier_.erase(stmt);
  }

  static void run(IRNode *node) {
    RemoveMatrixOfPtr pass;
    node->accept(&pass);
    pass.modifier_.modify_ir();
  }
};

namespace irpass {

void lower_matrix_ptr(IRNode *root) {
  TI_AUTO_PROF;

  GatherValidAOSGlobalPtrStmt gather_valid_aos_global_ptr_pass(root);

  LowerAOSGlobalPtrStmt lower_aos_global_ptr_stmt_pass(
      root, gather_valid_aos_global_ptr_pass.invalid_aos_global_ptr_stmts_);

  ScalarizeMatrixPtr scalarize_matrix_ptr_pass(root);
  LowerMatrixPtr::run(root);
  RemoveMatrixOfPtr::run(root);
}

}  // namespace irpass

}  // namespace taichi::lang
