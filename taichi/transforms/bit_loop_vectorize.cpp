// The bit-level loop vectorizer

#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/type_factory.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

class BitLoopVectorize : public IRVisitor {
 public:
  bool is_bit_vectorized;
  bool in_struct_for_loop;
  StructForStmt *loop_stmt;
  PrimitiveType *quant_array_physical_type;
  std::unordered_map<Stmt *, std::vector<Stmt *>> transformed_atomics;

  BitLoopVectorize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    is_bit_vectorized = false;
    in_struct_for_loop = false;
    loop_stmt = nullptr;
    quant_array_physical_type = nullptr;
  }

  void visit(Block *stmt_list) override {
    std::vector<Stmt *> statements;
    for (auto &stmt : stmt_list->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      stmt->accept(this);
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto ptr_type = stmt->src->ret_type->as<PointerType>();
    if (in_struct_for_loop && is_bit_vectorized) {
      if (ptr_type->get_pointee_type()->cast<QuantIntType>()) {
        // rewrite the previous GlobalPtrStmt's return type from *qit to
        // *phy_type
        auto ptr = stmt->src->cast<GlobalPtrStmt>();
        auto ptr_physical_type = TypeFactory::get_instance().get_pointer_type(
            quant_array_physical_type, false);
        DataType new_ret_type(ptr_physical_type);
        ptr->ret_type = new_ret_type;
        ptr->is_bit_vectorized = true;
        // check if j has offset
        if (ptr->indices.size() == 2) {
          auto diff = irpass::analysis::value_diff_loop_index(ptr->indices[1],
                                                              loop_stmt, 1);
          // TODO: temporarily we only support [j - 1] and [j + 1]
          //       the general case should be easy to implement
          if (diff.linear_related() && diff.certain() &&
              (diff.low == 1 || diff.low == -1)) {
            // construct ptr to x[i, j]
            auto indices = ptr->indices;
            indices[1] = loop_stmt->body->statements[1].get();
            auto base_ptr =
                std::make_unique<GlobalPtrStmt>(ptr->snode, indices);
            base_ptr->ret_type = new_ret_type;
            base_ptr->is_bit_vectorized = true;
            // load x[i, j](base)
            DataType load_data_type(quant_array_physical_type);
            auto load_base = std::make_unique<GlobalLoadStmt>(base_ptr.get());
            load_base->ret_type = load_data_type;
            // load x[i, j + 1](offsetted)
            // since we are doing vectorization, the actual data should be x[i,
            // j + vectorization_width]
            auto vectorization_width = data_type_bits(load_data_type);
            auto offset_constant =
                std::make_unique<ConstStmt>(TypedConstant(vectorization_width));
            auto offset_index_opcode =
                diff.low == -1 ? BinaryOpType::sub : BinaryOpType::add;
            auto offset_index = std::make_unique<BinaryOpStmt>(
                offset_index_opcode, indices[1], offset_constant.get());
            indices[1] = offset_index.get();
            auto offset_ptr =
                std::make_unique<GlobalPtrStmt>(ptr->snode, indices);
            offset_ptr->ret_type = new_ret_type;
            offset_ptr->is_bit_vectorized = true;
            auto load_offsetted =
                std::make_unique<GlobalLoadStmt>(offset_ptr.get());
            load_offsetted->ret_type = load_data_type;
            // create bit shift and bit and operations
            auto base_shift_offset =
                std::make_unique<ConstStmt>(TypedConstant(load_data_type, 1));
            auto base_shift_opcode =
                diff.low == -1 ? BinaryOpType::bit_shl : BinaryOpType::bit_sar;
            auto base_shift_op = std::make_unique<BinaryOpStmt>(
                base_shift_opcode, load_base.get(), base_shift_offset.get());

            auto offsetted_shift_offset = std::make_unique<ConstStmt>(
                TypedConstant(load_data_type, vectorization_width - 1));
            auto offsetted_shift_opcode =
                diff.low == -1 ? BinaryOpType::bit_sar : BinaryOpType::bit_shl;
            auto offsetted_shift_op = std::make_unique<BinaryOpStmt>(
                offsetted_shift_opcode, load_offsetted.get(),
                offsetted_shift_offset.get());

            auto or_op = std::make_unique<BinaryOpStmt>(
                BinaryOpType::bit_or, base_shift_op.get(),
                offsetted_shift_op.get());
            // modify IR
            auto offsetted_shift_op_p = offsetted_shift_op.get();
            stmt->insert_before_me(std::move(base_ptr));
            stmt->insert_before_me(std::move(load_base));
            stmt->insert_before_me(std::move(offset_constant));
            stmt->insert_before_me(std::move(offset_index));
            stmt->insert_before_me(std::move(offset_ptr));
            stmt->insert_before_me(std::move(load_offsetted));
            stmt->insert_before_me(std::move(base_shift_offset));
            stmt->insert_before_me(std::move(base_shift_op));
            stmt->insert_before_me(std::move(offsetted_shift_offset));
            stmt->insert_before_me(std::move(offsetted_shift_op));
            stmt->replace_usages_with(or_op.get());
            offsetted_shift_op_p->insert_after_me(std::move(or_op));
          }
        }
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    auto ptr_type = stmt->dest->ret_type->as<PointerType>();
    if (in_struct_for_loop && is_bit_vectorized) {
      if (ptr_type->get_pointee_type()->cast<QuantIntType>()) {
        // rewrite the previous GlobalPtrStmt's return type from *qit to
        // *phy_type
        auto ptr = stmt->dest->cast<GlobalPtrStmt>();
        auto ptr_physical_type = TypeFactory::get_instance().get_pointer_type(
            quant_array_physical_type, false);
        DataType new_ret_type(ptr_physical_type);
        ptr->ret_type = new_ret_type;
        ptr->is_bit_vectorized = true;
      }
    }
  }

  void visit(StructForStmt *stmt) override {
    if (stmt->snode->type != SNodeType::quant_array) {
      return;
    }
    bool old_is_bit_vectorized = is_bit_vectorized;
    is_bit_vectorized = stmt->is_bit_vectorized;
    in_struct_for_loop = true;
    loop_stmt = stmt;
    quant_array_physical_type = stmt->snode->physical_type;
    stmt->body->accept(this);
    is_bit_vectorized = old_is_bit_vectorized;
    in_struct_for_loop = false;
    loop_stmt = nullptr;
    quant_array_physical_type = nullptr;
  }

  void visit(BinaryOpStmt *stmt) override {
    // vectorize cmp_eq and bit_and between
    // vectorized data(local adder/array elems) and constant
    if (in_struct_for_loop && is_bit_vectorized) {
      if (stmt->op_type == BinaryOpType::bit_and) {
        // if the rhs is a bit vectorized stmt and lhs is a const 1
        // (usually generated by boolean expr), we simply replace
        // the stmt with its rhs
        int lhs_val = get_constant_value(stmt->lhs);
        if (lhs_val == 1) {
          if (auto rhs = stmt->rhs->cast<BinaryOpStmt>();
              rhs && rhs->is_bit_vectorized) {
            stmt->replace_usages_with(stmt->rhs);
          }
        }
      } else if (stmt->op_type == BinaryOpType::cmp_eq) {
        if (auto lhs = stmt->lhs->cast<GlobalLoadStmt>()) {
          // case 0: lhs is a vectorized global load from the quant array
          if (auto ptr = lhs->src->cast<GlobalPtrStmt>();
              ptr && ptr->is_bit_vectorized) {
            int32 rhs_val = get_constant_value(stmt->rhs);
            // TODO: we limit 1 for now, 0 should be easy to implement by a
            // bit_not on original bit pattern
            TI_ASSERT(rhs_val == 1);
            // cmp_eq with 1 yields the bit pattern itself

            // to pass CFG analysis and mark the stmt vectorized
            // create a dummy lhs + 0 here
            auto zero = std::make_unique<ConstStmt>(TypedConstant(0));
            auto add = std::make_unique<BinaryOpStmt>(BinaryOpType::add,
                                                      stmt->lhs, zero.get());
            add->is_bit_vectorized = true;
            // modify IR
            auto zero_p = zero.get();
            stmt->insert_before_me(std::move(zero));
            stmt->replace_usages_with(add.get());
            zero_p->insert_after_me(std::move(add));
          }
        } else if (auto lhs = stmt->lhs->cast<LocalLoadStmt>()) {
          // case 1: lhs is a local load from a local adder structure
          auto it = transformed_atomics.find(lhs->src);
          if (it != transformed_atomics.end()) {
            int32 rhs_val = get_constant_value(stmt->rhs);
            // TODO: we limit 2 and 3 for now, the other case should be
            // implement in a similar fashion
            TI_ASSERT(rhs_val == 2 || rhs_val == 3);
            // 010 and 011 respectively
            auto &buffer_vec = it->second;
            Stmt *a = buffer_vec[0], *b = buffer_vec[1], *c = buffer_vec[2];
            // load all three buffers
            auto load_a = std::make_unique<LocalLoadStmt>(a);
            auto load_b = std::make_unique<LocalLoadStmt>(b);
            auto load_c = std::make_unique<LocalLoadStmt>(c);
            // compute not_a first
            auto not_a = std::make_unique<UnaryOpStmt>(UnaryOpType::bit_not,
                                                       load_a.get());
            // b should always be itself so do nothing
            // compute not_c
            auto not_c = std::make_unique<UnaryOpStmt>(UnaryOpType::bit_not,
                                                       load_c.get());
            // bit_and all three patterns
            auto and_a_b = std::make_unique<BinaryOpStmt>(
                BinaryOpType::bit_and, not_a.get(), load_b.get());
            auto and_b_c = std::make_unique<BinaryOpStmt>(
                BinaryOpType::bit_and, and_a_b.get(),
                rhs_val == 2 ? (Stmt *)(not_c.get()) : (Stmt *)(load_c.get()));
            // mark the last stmt as vectorized
            and_b_c->is_bit_vectorized = true;
            // modify IR
            auto and_a_b_p = and_a_b.get();
            stmt->insert_before_me(std::move(load_a));
            stmt->insert_before_me(std::move(load_b));
            stmt->insert_before_me(std::move(load_c));
            stmt->insert_before_me(std::move(not_a));
            stmt->insert_before_me(std::move(not_c));
            stmt->insert_before_me(std::move(and_a_b));
            stmt->replace_usages_with(and_b_c.get());
            and_a_b_p->insert_after_me(std::move(and_b_c));
          }
        }
      }
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    DataType dt(quant_array_physical_type);
    if (in_struct_for_loop && is_bit_vectorized &&
        stmt->op_type == AtomicOpType::add) {
      auto it = transformed_atomics.find(stmt->dest);
      // process a transformed atomic stmt
      if (it != transformed_atomics.end()) {
        auto &buffer_vec = it->second;
        transform_atomic_add(buffer_vec, stmt, dt);
      } else {
        // alloc three buffers a, b, c
        auto alloc_a = std::make_unique<AllocaStmt>(dt);
        auto alloc_b = std::make_unique<AllocaStmt>(dt);
        auto alloc_c = std::make_unique<AllocaStmt>(dt);
        std::vector<Stmt *> buffer_vec{alloc_a.get(), alloc_b.get(),
                                       alloc_c.get()};
        transformed_atomics[stmt->dest] = buffer_vec;
        // modify IR
        stmt->insert_before_me(std::move(alloc_a));
        stmt->insert_before_me(std::move(alloc_b));
        stmt->insert_before_me(std::move(alloc_c));
        transform_atomic_add(buffer_vec, stmt, dt);
      }
    }
  }

  static void run(IRNode *node) {
    BitLoopVectorize inst;
    node->accept(&inst);
  }

 private:
  void transform_atomic_add(const std::vector<Stmt *> &buffer_vec,
                            AtomicOpStmt *stmt,
                            DataType &dt) {
    // To transform an atomic add on a vectorized subarray of a quant array,
    // we use a local adder with three buffers(*a*,*b*,*c*) of the same physical
    // type of the original quant array. Each bit in *a* represents the highest
    // bit of the result, while *b* for the second bit and *c* for the lowest
    // bit To add *d* to the subarray, we do bit_xor and bit_and to compute the
    // sum and the carry
    Stmt *a = buffer_vec[0], *b = buffer_vec[1], *c = buffer_vec[2];
    auto load_c = std::make_unique<LocalLoadStmt>(c);
    auto carry_c = std::make_unique<BinaryOpStmt>(BinaryOpType::bit_and,
                                                  load_c.get(), stmt->val);
    auto sum_c =
        std::make_unique<AtomicOpStmt>(AtomicOpType::bit_xor, c, stmt->val);
    auto load_b = std::make_unique<LocalLoadStmt>(b);
    auto carry_b = std::make_unique<BinaryOpStmt>(BinaryOpType::bit_and,
                                                  load_b.get(), carry_c.get());
    auto sum_b =
        std::make_unique<AtomicOpStmt>(AtomicOpType::bit_xor, b, carry_c.get());
    // for a, we do not need to compute its carry
    auto sum_a =
        std::make_unique<AtomicOpStmt>(AtomicOpType::bit_xor, a, carry_b.get());
    // modify IR
    stmt->insert_before_me(std::move(load_c));
    stmt->insert_before_me(std::move(carry_c));
    stmt->insert_before_me(std::move(sum_c));
    stmt->insert_before_me(std::move(load_b));
    stmt->insert_before_me(std::move(carry_b));
    stmt->insert_before_me(std::move(sum_b));
    stmt->insert_before_me(std::move(sum_a));
    // there is no need to replace the stmt here as we
    // will replace it manually later
  }

  int32 get_constant_value(Stmt *stmt) {
    int32 val = -1;
    // the stmt could be a cast stmt
    if (auto cast_stmt = stmt->cast<UnaryOpStmt>();
        cast_stmt && cast_stmt->is_cast() &&
        cast_stmt->op_type == UnaryOpType::cast_value) {
      stmt = cast_stmt->operand;
    }
    if (auto constant_stmt = stmt->cast<ConstStmt>();
        constant_stmt &&
        constant_stmt->val.dt->is_primitive(PrimitiveTypeID::i32)) {
      val = constant_stmt->val.val_i32;
    }
    return val;
  }
};

namespace irpass {

void bit_loop_vectorize(IRNode *root) {
  TI_AUTO_PROF;
  BitLoopVectorize::run(root);
  die(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
