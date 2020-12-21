// The bit-level loop vectorizer

#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/type_factory.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class BitLoopVectorize : public IRVisitor {
 public:
  int bit_vectorize;
  bool in_struct_for_loop;
  PrimitiveType* bit_array_physical_type;

  BitLoopVectorize() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    bit_vectorize = 1;
    in_struct_for_loop = false;
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
        // rewrite the previous GlobalPtrStmt's return type from *cit to *phy_type
        auto ptr = stmt->ptr->cast<GlobalPtrStmt>();
        auto ptr_physical_type = TypeFactory::get_instance().get_pointer_type(bit_array_physical_type, false);
        DataType new_ret_type(ptr_physical_type);
        ptr->ret_type = new_ret_type;
        ptr->is_bit_vectorized = true;
        // TODO: Do we need to explicitly make the load stmt's return type same as physical type
        //       for now, this seems to hold under the demo code
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    auto ptr_type = stmt->ptr->ret_type->as<PointerType>();
    if (in_struct_for_loop && bit_vectorize != 1) {
      if (auto cit = ptr_type->get_pointee_type()->cast<CustomIntType>()) {
        // rewrite the previous GlobalPtrStmt's return type from *cit to *phy_type
        auto ptr = stmt->ptr->cast<GlobalPtrStmt>();
        auto ptr_physical_type = TypeFactory::get_instance().get_pointer_type(bit_array_physical_type, false);
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
    bit_array_physical_type = stmt->snode->physical_type;
    stmt->body->accept(this);
    bit_vectorize = old_bit_vectorize;
    in_struct_for_loop = false;
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
