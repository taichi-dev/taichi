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
  int bit_vectorize;
  bool in_struct_for_loop;
  StructForStmt* loop_stmt;
  PrimitiveType *bit_array_physical_type;

  BitLoopVectorize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    bit_vectorize = 1;
    in_struct_for_loop = false;
    loop_stmt = nullptr;
    bit_array_physical_type = nullptr;
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
    auto ptr_type = stmt->ptr->ret_type->as<PointerType>();
    if (in_struct_for_loop && bit_vectorize != 1) {
      if (auto cit = ptr_type->get_pointee_type()->cast<CustomIntType>()) {
        // rewrite the previous GlobalPtrStmt's return type from *cit to
        // *phy_type
        auto ptr = stmt->ptr->cast<GlobalPtrStmt>();
        auto ptr_physical_type = TypeFactory::get_instance().get_pointer_type(
            bit_array_physical_type, false);
        DataType new_ret_type(ptr_physical_type);
        ptr->ret_type = new_ret_type;
        ptr->is_bit_vectorized = true;
        // check if j has offset
        if (ptr->indices.size() == 2) {
          auto diff = irpass::analysis::value_diff_loop_index(ptr->indices[1], loop_stmt, 1);
          // TODO: temporarily we only support [j - 1] and [j + 1]
            //       the general case should be easy to implement
          if (diff.linear_related() && diff.high - diff.low == 1 && (diff.low == 1 || diff.low == -1)) {
            // construct ptr to x[i, j]
            auto indices = ptr->indices;
            indices[1] = loop_stmt->body->statements[1].get();
            auto base_ptr = std::make_unique<GlobalPtrStmt>(ptr->snodes, indices);
            base_ptr->ret_type = new_ret_type;
            base_ptr->is_bit_vectorized = true;
            // load x[i, j](base) and x[i, j + 1](offsetted)
            DataType load_data_type(bit_array_physical_type);
            auto load_base = std::make_unique<GlobalLoadStmt>(base_ptr.get());
            load_base->ret_type = load_data_type;
            auto load_offsetted = std::make_unique<GlobalLoadStmt>(ptr);
            load_offsetted->ret_type = load_data_type;
            // create bit shift and bit and operations
            auto base_shift_offset = std::make_unique<ConstStmt>(TypedConstant(1));
            auto base_shift_opcode = diff.low == -1 ? BinaryOpType::bit_shl : BinaryOpType::bit_sar;
            auto base_shift_op = std::make_unique<BinaryOpStmt>(base_shift_opcode, load_base.get(), base_shift_offset.get());

            auto offsetted_shift_offset = std::make_unique<ConstStmt>(TypedConstant(bit_vectorize - 1));
            auto offsetted_shift_opcode = diff.low == -1 ? BinaryOpType::bit_sar : BinaryOpType::bit_shl;
            auto offsetted_shift_op = std::make_unique<BinaryOpStmt>(offsetted_shift_opcode, load_offsetted.get(), offsetted_shift_offset.get());

            auto or_op = std::make_unique<BinaryOpStmt>(BinaryOpType::bit_or, base_shift_op.get(), offsetted_shift_op.get());
            // modify IR
            auto offsetted_shift_op_p = offsetted_shift_op.get();
            stmt->insert_before_me(std::move(base_ptr));
            stmt->insert_before_me(std::move(load_base));
            stmt->insert_before_me(std::move(load_offsetted));
            stmt->insert_before_me(std::move(base_shift_offset));
            stmt->insert_before_me(std::move(std::move(base_shift_op)));
            stmt->insert_before_me(std::move(offsetted_shift_offset));
            stmt->insert_before_me(std::move(std::move(offsetted_shift_op)));
            stmt->replace_with(or_op.get());
            offsetted_shift_op_p->insert_after_me(std::move(or_op));
          }
        }
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    auto ptr_type = stmt->ptr->ret_type->as<PointerType>();
    if (in_struct_for_loop && bit_vectorize != 1) {
      if (auto cit = ptr_type->get_pointee_type()->cast<CustomIntType>()) {
        // rewrite the previous GlobalPtrStmt's return type from *cit to
        // *phy_type
        auto ptr = stmt->ptr->cast<GlobalPtrStmt>();
        auto ptr_physical_type = TypeFactory::get_instance().get_pointer_type(
            bit_array_physical_type, false);
        DataType new_ret_type(ptr_physical_type);
        ptr->ret_type = new_ret_type;
        ptr->is_bit_vectorized = true;
      }
    }
  }

  void visit(StructForStmt *stmt) override {
    if (stmt->snode->type != SNodeType::bit_array) {
      return;
    }
    int old_bit_vectorize = bit_vectorize;
    bit_vectorize = stmt->bit_vectorize;
    in_struct_for_loop = true;
    loop_stmt = stmt;
    bit_array_physical_type = stmt->snode->physical_type;
    stmt->body->accept(this);
    bit_vectorize = old_bit_vectorize;
    in_struct_for_loop = false;
    loop_stmt = nullptr;
    bit_array_physical_type = nullptr;
  }

  static void run(IRNode *node) {
    BitLoopVectorize inst;
    node->accept(&inst);
  }
};

namespace irpass {

void bit_loop_vectorize(IRNode *root) {
  TI_AUTO_PROF;
  return BitLoopVectorize::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
