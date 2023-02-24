#include <variant>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {

class Half2VectorizationAnalyzer : public BasicStmtVisitor {
 public:
  explicit Half2VectorizationAnalyzer(IRNode *node) {
    node->accept(this);
  }

  std::unordered_set<AtomicOpStmt *> should_remove;
  std::map<AtomicOpStmt *, AtomicOpStmt *>
      should_replace_extern;  // self: other
  std::map<AtomicOpStmt *, AtomicOpStmt *>
      should_replace_global_temp;  // self: other
  std::map<AtomicOpStmt *, AtomicOpStmt *>
      should_replace_get_ch;  // self: other

  /* Auto vectorization for "half2" - TensorType(fp16, 2)

    [Ndarray] Implemented
    Condition: Two ExternalPtrStmts having the same base_ptr & inner most
    indices are 0 and 1 Before: i32 const_0 = ConstStmt(0) i32 const_1 =
    ConstStmt(1)

        f16* ptr_0 = ExternalPtrStmt(arg, [$1, const_0])
        f16* ptr_1 = ExternalPtrStmt(arg, [$1, const_1])

        f16 old_val0 = AtomicStmt(ptr_0, $7)
        f16 old_val1 = AtomicStmt(ptr_1, $8)

    After:
        TensorType(2, f16) val = MatrixInitStmt([$7, $8])

        TensorType(2, f16)* ptr = ExternalPtrStmt(arg, [$1])
        TensorType(2, f16) old_val = AtomicStmt(ptr, val)

        TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
        StoreStmt(old_val, old_val_alloc)

        f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
        f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

        alloca_stmt0->replace_all_usages_with(old_val0);
        alloca_stmt1->replace_all_usages_with(old_val1);

    [GlobalTemp] Implemented
    Condition: Two GlobalTempStmts' offsets diff is 2
    Before:
        f16* ptr_0 = GlobalTempStmt(offset0)
        f16* ptr_1 = GlobalTempStmt(offset0 + 2)

        f16 old_val0 = AtomicStmt(ptr_0, $7)
        f16 old_val1 = AtomicStmt(ptr_1, $8)

    After:
        TensorType(2, f16) val = MatrixInitStmt([$7, $8])

        TensorType(2, f16)* ptr = GlobalTempStmt(offset0)
        TensorType(2, f16) old_val = AtomicStmt(ptr, val)

        TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
        StoreStmt(old_val, old_val_alloc)

        f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
        f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

        alloca_stmt0->replace_all_usages_with(old_val0);
        alloca_stmt1->replace_all_usages_with(old_val1);

    [Alloca]
    Before:
        f16* ptr_0 = AllocaStmt(fp16)
        f16* ptr_1 = AllocaStmt(fp16)

        f16 old_val0 = AtomicStmt(ptr_0, $7)
        f16 old_val1 = AtomicStmt(ptr_1, $8)

    After:
        TensorType(2, f16)* ptr = AllocaStmt(TensorType(2, f16))
        f16* ptr_0 = MatrixPtrStmt(ptr, 0)
        f16* ptr_1 = MatrixPtrStmt(ptr, 1)
        alloca_stmt0.replace_all_usages_with(ptr_0)
        alloca_stmt1.replace_all_usages_with(ptr_1)

        TensorType(2, f16) val = MatrixInitStmt([$7, $8])

        TensorType(2, f16) old_val = AtomicStmt(ptr, val)

        TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
        StoreStmt(old_val, old_val_alloc)

        f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
        f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

        alloca_stmt0->replace_all_usages_with(old_val0);
        alloca_stmt1->replace_all_usages_with(old_val1);

    [Field] Implemented
    Condition: Two GetChStmts having the same container & chids are 0 and 1
    Before:
        gen* container = SNodeLookupStmt(...)

        fp16* ptr_0 = GetChStmt(container, 0)
        fp16* ptr_1 = GetChStmt(container, 1)

        f16 old_val0 = AtomicStmt(ptr_0, $7)
        f16 old_val1 = AtomicStmt(ptr_1, $8)

    After:
        TensorType(2, f16) val = MatrixInitStmt([$7, $8])

        TensorType(2, f16)* ptr = GetChStmt(container, 0)
        TensorType(2, f16) old_val = AtomicStmt(ptr, val)

        TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
        StoreStmt(old_val, old_val_alloc)

        f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
        f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

        alloca_stmt0->replace_all_usages_with(old_val0);
        alloca_stmt1->replace_all_usages_with(old_val1);

  */
  void visit(AtomicOpStmt *stmt) override {
    // opt-out
    if (stmt->ret_type != PrimitiveType::f16) {
      return;
    }

    if (stmt->op_type != AtomicOpType::add) {
      return;
    }

    int stmt_type;  // 0: ExternalPtrStmt, 1: GlobalTemporaryStmt, 2: GetChStmt
    if (stmt->dest->is<ExternalPtrStmt>() &&
        stmt->dest->cast<ExternalPtrStmt>()->indices.back()->is<ConstStmt>()) {
      stmt_type = 0;
    } else if (stmt->dest->is<GlobalTemporaryStmt>()) {
      stmt_type = 1;
    } else if (stmt->dest->is<GetChStmt>()) {
      stmt_type = 2;
    } else {
      return;
    }

    for (auto iter : recorded_atomic_ops_) {
      auto *atomic_op = iter;
      if (stmt_type == 0) {
        auto *self_extern_stmt = stmt->dest->cast<ExternalPtrStmt>();
        auto *other_extern_stmt = atomic_op->dest->cast<ExternalPtrStmt>();

        if (self_extern_stmt->base_ptr != other_extern_stmt->base_ptr) {
          continue;
        }

        std::vector<Stmt *> self_extern_indices = self_extern_stmt->indices;
        std::vector<Stmt *> other_extern_indices = other_extern_stmt->indices;
        if (self_extern_indices.size() != other_extern_indices.size()) {
          continue;
        }

        if (self_extern_indices.back()->cast<ConstStmt>()->val.val_int32() +
                other_extern_indices.back()
                    ->cast<ConstStmt>()
                    ->val.val_int32() ==
            1) {
          // Found pair
          recorded_atomic_ops_.erase(iter);

          if (self_extern_indices.back()->cast<ConstStmt>()->val.val_int32() ==
              0) {
            should_remove.insert(atomic_op);
            should_replace_extern[stmt] = atomic_op;
          } else {
            should_remove.insert(stmt);
            should_replace_extern[atomic_op] = stmt;
          }
        }
      }

      if (stmt_type == 1) {
        auto *self_global_temp_stmt = stmt->dest->cast<GlobalTemporaryStmt>();
        auto *other_global_temp_stmt =
            atomic_op->dest->cast<GlobalTemporaryStmt>();

        int offset_diff =
            std::abs(static_cast<int>(self_global_temp_stmt->offset) -
                     static_cast<int>(other_global_temp_stmt->offset));
        if (offset_diff == 2) {
          // Found pair
          recorded_atomic_ops_.erase(iter);

          if (self_global_temp_stmt->offset < other_global_temp_stmt->offset) {
            should_remove.insert(atomic_op);
            should_replace_global_temp[stmt] = atomic_op;
          } else {
            should_remove.insert(stmt);
            should_replace_global_temp[atomic_op] = stmt;
          }
        }
      }

      if (stmt_type == 2) {
        auto *self_get_ch_stmt = stmt->dest->cast<GetChStmt>();
        auto *other_get_ch_stmt = atomic_op->dest->cast<GetChStmt>();

        if (self_get_ch_stmt->input_ptr != other_get_ch_stmt->input_ptr) {
          continue;
        }

        if ((self_get_ch_stmt->chid == 0 && other_get_ch_stmt->chid == 1) ||
            (self_get_ch_stmt->chid == 1 && other_get_ch_stmt->chid == 0)) {
          // Found pair
          recorded_atomic_ops_.erase(iter);

          if (self_get_ch_stmt->chid == 0) {
            should_remove.insert(atomic_op);
            should_replace_get_ch[stmt] = atomic_op;
          } else {
            should_remove.insert(stmt);
            should_replace_get_ch[atomic_op] = stmt;
          }
        }
      }
    }

    recorded_atomic_ops_.insert(stmt);
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->tls_prologue)
      stmt->tls_prologue->accept(this);

    if (stmt->body) {
      stmt->body->accept(this);
    }

    if (stmt->tls_epilogue)
      stmt->tls_epilogue->accept(this);
  }

 private:
  std::unordered_set<AtomicOpStmt *> recorded_atomic_ops_;
  using BasicStmtVisitor::visit;
};

class Half2Vectorize : public BasicStmtVisitor {
 public:
  DelayedIRModifier delayed_modifier_;

  explicit Half2Vectorize(
      IRNode *node,
      const std::unordered_set<AtomicOpStmt *> &should_remove,
      const std::map<AtomicOpStmt *, AtomicOpStmt *> &should_replace_extern,
      const std::map<AtomicOpStmt *, AtomicOpStmt *>
          &should_replace_global_temp,
      const std::map<AtomicOpStmt *, AtomicOpStmt *> &should_replace_get_ch) {
    this->should_remove = should_remove;
    this->should_replace_extern = should_replace_extern;
    this->should_replace_global_temp = should_replace_global_temp;
    this->should_replace_get_ch = should_replace_get_ch;

    node->accept(this);

    delayed_modifier_.modify_ir();
  }

  std::unordered_set<AtomicOpStmt *> should_remove;
  std::map<AtomicOpStmt *, AtomicOpStmt *>
      should_replace_extern;  // self: other
  std::map<AtomicOpStmt *, AtomicOpStmt *>
      should_replace_global_temp;  // self: other
  std::map<AtomicOpStmt *, AtomicOpStmt *>
      should_replace_get_ch;  // self: other

  void visit(AtomicOpStmt *stmt) override {
    if (should_remove.find(stmt) != should_remove.end()) {
      delayed_modifier_.erase(stmt);
      return;
    }

    if (should_replace_extern.find(stmt) != should_replace_extern.end()) {
      auto *self_extern_stmt = stmt->dest->cast<ExternalPtrStmt>();
      auto *self_ptr = self_extern_stmt->base_ptr;
      std::vector<Stmt *> self_indices = self_extern_stmt->indices;
      auto *self_val = stmt->val;

      AtomicOpStmt *other_stmt = should_replace_extern[stmt];
      auto *other_extern_stmt = other_stmt->dest->cast<ExternalPtrStmt>();
      std::vector<Stmt *> other_indices = other_extern_stmt->indices;
      auto *other_val = other_stmt->val;

      // Create MatrixInitStmt
      std::vector<Stmt *> matrix_init_values;
      matrix_init_values.push_back(self_val);
      matrix_init_values.push_back(other_val);

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      auto tensor_type =
          TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::f16);
      matrix_init_stmt->ret_type = tensor_type;

      // Create ExternalPtrStmt
      std::vector<Stmt *> new_indices = self_indices;
      new_indices.pop_back();  // Remove last index

      std::vector<int> element_shape = {2};
      int element_dim = -1;
      auto new_extern_stmt = std::make_unique<ExternalPtrStmt>(
          self_ptr, new_indices, element_shape, element_dim);
      new_extern_stmt->overrided_dtype = true;
      new_extern_stmt->ret_type = tensor_type;
      new_extern_stmt->ret_type.set_is_pointer(true);

      // Create AtomicStmt
      auto new_atomic_stmt = std::make_unique<AtomicOpStmt>(
          AtomicOpType::add, new_extern_stmt.get(), matrix_init_stmt.get());
      new_atomic_stmt->ret_type = tensor_type;

      // Create AllocaStmt
      auto new_alloc_stmt =
          std::make_unique<AllocaStmt>(matrix_init_stmt->ret_type);
      new_alloc_stmt->ret_type = tensor_type;
      new_alloc_stmt->ret_type.set_is_pointer(true);

      // Create StoreStmt
      auto new_store_stmt = std::make_unique<LocalStoreStmt>(
          new_alloc_stmt.get(), new_atomic_stmt.get());

      // Create MatrixPtrStmt
      auto const_0 = std::make_unique<ConstStmt>(TypedConstant(0));
      auto const_1 = std::make_unique<ConstStmt>(TypedConstant(1));
      const_0->ret_type = PrimitiveType::i32;
      const_1->ret_type = PrimitiveType::i32;

      auto new_matrix_ptr_stmt0 =
          std::make_unique<MatrixPtrStmt>(new_alloc_stmt.get(), const_0.get());
      auto new_matrix_ptr_stmt1 =
          std::make_unique<MatrixPtrStmt>(new_alloc_stmt.get(), const_1.get());
      new_matrix_ptr_stmt0->ret_type = PrimitiveType::f16;
      new_matrix_ptr_stmt1->ret_type = PrimitiveType::f16;

      if (other_indices.back()->cast<ConstStmt>()->val.val_int32() == 1) {
        other_stmt->replace_usages_with(new_matrix_ptr_stmt1.get());
        stmt->replace_usages_with(new_matrix_ptr_stmt0.get());
      } else {
        other_stmt->replace_usages_with(new_matrix_ptr_stmt0.get());
        stmt->replace_usages_with(new_matrix_ptr_stmt1.get());
      }

      delayed_modifier_.insert_before(stmt, std::move(const_0));
      delayed_modifier_.insert_before(stmt, std::move(const_1));
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_extern_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_atomic_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_alloc_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_store_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_matrix_ptr_stmt0));
      delayed_modifier_.insert_before(stmt, std::move(new_matrix_ptr_stmt1));

      delayed_modifier_.erase(stmt);
    }

    if (should_replace_global_temp.find(stmt) !=
        should_replace_global_temp.end()) {
      auto *self_global_temp_stmt = stmt->dest->cast<GlobalTemporaryStmt>();
      size_t self_offset = self_global_temp_stmt->offset;
      auto *self_val = stmt->val;

      AtomicOpStmt *other_stmt = should_replace_global_temp[stmt];
      size_t other_offset =
          other_stmt->dest->cast<GlobalTemporaryStmt>()->offset;
      auto *other_val = other_stmt->val;

      // Create MatrixInitStmt
      std::vector<Stmt *> matrix_init_values;
      matrix_init_values.push_back(self_val);
      matrix_init_values.push_back(other_val);

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      auto tensor_type =
          TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::f16);
      matrix_init_stmt->ret_type = tensor_type;

      // Create GlobalTempStmt
      auto new_global_temp_stmt = std::make_unique<GlobalTemporaryStmt>(
          std::min(self_offset, other_offset), tensor_type);
      new_global_temp_stmt->ret_type = tensor_type;
      new_global_temp_stmt->ret_type.set_is_pointer(true);

      // Create AtomicStmt
      auto new_atomic_stmt = std::make_unique<AtomicOpStmt>(
          AtomicOpType::add, new_global_temp_stmt.get(),
          matrix_init_stmt.get());
      new_atomic_stmt->ret_type = tensor_type;

      // Create AllocaStmt
      auto new_alloc_stmt =
          std::make_unique<AllocaStmt>(matrix_init_stmt->ret_type);
      new_alloc_stmt->ret_type = tensor_type;
      new_alloc_stmt->ret_type.set_is_pointer(true);

      // Create StoreStmt
      auto new_store_stmt = std::make_unique<LocalStoreStmt>(
          new_alloc_stmt.get(), new_atomic_stmt.get());

      // Create MatrixPtrStmt
      auto const_0 = std::make_unique<ConstStmt>(TypedConstant(0));
      auto const_1 = std::make_unique<ConstStmt>(TypedConstant(1));
      const_0->ret_type = PrimitiveType::i32;
      const_1->ret_type = PrimitiveType::i32;

      auto new_matrix_ptr_stmt0 =
          std::make_unique<MatrixPtrStmt>(new_alloc_stmt.get(), const_0.get());
      auto new_matrix_ptr_stmt1 =
          std::make_unique<MatrixPtrStmt>(new_alloc_stmt.get(), const_1.get());
      new_matrix_ptr_stmt0->ret_type = PrimitiveType::f16;
      new_matrix_ptr_stmt1->ret_type = PrimitiveType::f16;

      if (other_offset > self_offset) {
        other_stmt->replace_usages_with(new_matrix_ptr_stmt1.get());
        stmt->replace_usages_with(new_matrix_ptr_stmt0.get());
      } else {
        other_stmt->replace_usages_with(new_matrix_ptr_stmt0.get());
        stmt->replace_usages_with(new_matrix_ptr_stmt1.get());
      }

      delayed_modifier_.insert_before(stmt, std::move(const_0));
      delayed_modifier_.insert_before(stmt, std::move(const_1));
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_global_temp_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_atomic_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_alloc_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_store_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_matrix_ptr_stmt0));
      delayed_modifier_.insert_before(stmt, std::move(new_matrix_ptr_stmt1));

      delayed_modifier_.erase(stmt);
    }

    if (should_replace_get_ch.find(stmt) != should_replace_get_ch.end()) {
      auto *self_get_ch_stmt = stmt->dest->cast<GetChStmt>();
      Stmt *self_input_ptr = self_get_ch_stmt->input_ptr;
      size_t self_chid = self_get_ch_stmt->chid;
      auto *self_val = stmt->val;

      AtomicOpStmt *other_stmt = should_replace_get_ch[stmt];
      auto *other_val = other_stmt->val;

      // Create MatrixInitStmt
      std::vector<Stmt *> matrix_init_values;
      matrix_init_values.push_back(self_val);
      matrix_init_values.push_back(other_val);

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      auto tensor_type =
          TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::f16);
      matrix_init_stmt->ret_type = tensor_type;

      // Create GlobalTempStmt
      auto new_get_ch_stmt = std::make_unique<GetChStmt>(
          self_input_ptr, self_chid, self_get_ch_stmt->is_bit_vectorized);
      new_get_ch_stmt->input_snode = self_get_ch_stmt->input_snode;
      new_get_ch_stmt->output_snode = self_get_ch_stmt->output_snode;
      new_get_ch_stmt->ret_type = tensor_type;
      new_get_ch_stmt->ret_type.set_is_pointer(true);
      new_get_ch_stmt->overrided_dtype = true;

      // Create AtomicStmt
      auto new_atomic_stmt = std::make_unique<AtomicOpStmt>(
          AtomicOpType::add, new_get_ch_stmt.get(), matrix_init_stmt.get());
      new_atomic_stmt->ret_type = tensor_type;

      // Create AllocaStmt
      auto new_alloc_stmt =
          std::make_unique<AllocaStmt>(matrix_init_stmt->ret_type);
      new_alloc_stmt->ret_type = tensor_type;
      new_alloc_stmt->ret_type.set_is_pointer(true);

      // Create StoreStmt
      auto new_store_stmt = std::make_unique<LocalStoreStmt>(
          new_alloc_stmt.get(), new_atomic_stmt.get());

      // Create MatrixPtrStmt
      auto const_0 = std::make_unique<ConstStmt>(TypedConstant(0));
      auto const_1 = std::make_unique<ConstStmt>(TypedConstant(1));
      const_0->ret_type = PrimitiveType::i32;
      const_1->ret_type = PrimitiveType::i32;

      auto new_matrix_ptr_stmt0 =
          std::make_unique<MatrixPtrStmt>(new_alloc_stmt.get(), const_0.get());
      auto new_matrix_ptr_stmt1 =
          std::make_unique<MatrixPtrStmt>(new_alloc_stmt.get(), const_1.get());
      new_matrix_ptr_stmt0->ret_type = PrimitiveType::f16;
      new_matrix_ptr_stmt1->ret_type = PrimitiveType::f16;

      stmt->replace_usages_with(new_matrix_ptr_stmt0.get());
      other_stmt->replace_usages_with(new_matrix_ptr_stmt1.get());

      delayed_modifier_.insert_before(stmt, std::move(const_0));
      delayed_modifier_.insert_before(stmt, std::move(const_1));
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_get_ch_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_atomic_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_alloc_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_store_stmt));
      delayed_modifier_.insert_before(stmt, std::move(new_matrix_ptr_stmt0));
      delayed_modifier_.insert_before(stmt, std::move(new_matrix_ptr_stmt1));

      delayed_modifier_.erase(stmt);
    }
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->tls_prologue)
      stmt->tls_prologue->accept(this);

    if (stmt->body) {
      stmt->body->accept(this);
    }

    if (stmt->tls_epilogue)
      stmt->tls_epilogue->accept(this);
  }

 private:
  using BasicStmtVisitor::visit;
};

namespace irpass {

void vectorize_half2(IRNode *root) {
  TI_AUTO_PROF;

  Half2VectorizationAnalyzer analyzer(root);

  Half2Vectorize vectorize_pass(
      root, analyzer.should_remove, analyzer.should_replace_extern,
      analyzer.should_replace_global_temp, analyzer.should_replace_get_ch);
}

}  // namespace irpass

}  // namespace taichi::lang
