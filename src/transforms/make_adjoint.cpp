#include <typeinfo>
#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class MakeAdjoint : public IRVisitor {
 public:
  Block *current_block;

  MakeAdjoint() {
    current_block = nullptr;
  }

  static void run(IRNode *node) {
    auto p = MakeAdjoint();
    node->accept(&p);
  }

  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    std::reverse(statements.begin(), statements.end());  // reverse-mode AD...
    for (auto stmt : statements) {
      current_block = block;
      stmt->accept(this);
    }
  }

  void insert_back(std::unique_ptr<Stmt> &&stmt) {
    current_block->insert(std::move(stmt), current_block->statements.size());
  }

  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = alloc(primal);
    TC_ASSERT(alloca_->is<AllocaStmt>());
    auto alloca = alloca_->as<AllocaStmt>();
    TC_ASSERT(alloca->width() == 1);
    auto local_load = Stmt::make<LocalLoadStmt>(LocalAddress(alloca, 0));
    if (value->is<AllocaStmt>()) {
      auto value_load = Stmt::make<LocalLoadStmt>(
          LocalAddress(value->as<AllocaStmt>(), 0));
      value = value_load.get();
      insert_back(std::move(value_load));
    }
    auto add =
        Stmt::make<BinaryOpStmt>(BinaryOpType::add, local_load.get(), value);
    auto local_store = Stmt::make<LocalStoreStmt>(alloca, add.get());
    insert_back(std::move(local_load));
    insert_back(std::move(add));
    insert_back(std::move(local_store));
  }

  Stmt *alloc(Stmt *stmt) {
    if (stmt->adjoint == nullptr) {
      // create the alloca
      auto alloca = Stmt::make<AllocaStmt>(1, DataType::unknown);
      stmt->adjoint = alloca.get();
      alloca->ret_type = stmt->ret_type;
      current_block->insert(std::move(alloca), 0);
    }
    return stmt->adjoint;
  }

  void visit(AllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(UnaryOpStmt *stmt) override {
  }

  void visit(BinaryOpStmt *bin) override {
    if (bin->op_type == BinaryOpType::add) {
      accumulate(bin->lhs, alloc(bin));
      accumulate(bin->rhs, alloc(bin));
    } else if (bin->op_type == BinaryOpType::mul){
      auto lmul = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, alloc(bin), bin->rhs);
      auto rmul = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, alloc(bin), bin->lhs);
      auto lptr = lmul.get();
      auto rptr = rmul.get();
      insert_back(std::move(lmul));
      insert_back(std::move(rmul));
      accumulate(bin->lhs, lptr);
      accumulate(bin->rhs, rptr);
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(IfStmt *if_stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(PrintStmt *print_stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(ConstStmt *const_stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(WhileControlStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(WhileStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(RangeForStmt *for_stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(StructForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(GlobalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(LocalLoadStmt *stmt) override {
    // do nothing
    TC_WARN("needs impl when loading something other loop var");
  }

  void visit(LocalStoreStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to adjoint
    GlobalPtrStmt *ptr = stmt->ptr->as<GlobalPtrStmt>();
    TC_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    TC_ASSERT(snodes[0]->grad != nullptr);
    snodes[0] = snodes[0]->grad;
    auto adjoint_ptr = Stmt::make<GlobalPtrStmt>(snodes, ptr->indices);
    auto adjoint_store = Stmt::make<GlobalStoreStmt>(adjoint_ptr.get(), alloc(stmt));
    auto adjoint_store_ptr = adjoint_store.get();
    insert_back(std::move(adjoint_ptr));
    insert_back(std::move(adjoint_store));
  }

  void visit(GlobalStoreStmt *stmt) override {
    // erase and replace with global load adjoint
    GlobalPtrStmt *ptr = stmt->ptr->as<GlobalPtrStmt>();
    TC_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    TC_ASSERT(snodes[0]->grad != nullptr);
    snodes[0] = snodes[0]->grad;
    auto adjoint_ptr = Stmt::make<GlobalPtrStmt>(snodes, ptr->indices);
    auto adjoint_load = Stmt::make<GlobalLoadStmt>(adjoint_ptr.get());
    auto adjoint_load_ptr = adjoint_load.get();
    insert_back(std::move(adjoint_ptr));
    insert_back(std::move(adjoint_load));
    accumulate(stmt->data, adjoint_load_ptr);
    stmt->parent->erase(stmt);
  }

  void visit(ElementShuffleStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(RangeAssumptionStmt *stmt) override {
    // do nothing
  }
};

namespace irpass {

void make_adjoint(IRNode *root) {
  MakeAdjoint::run(root);
  print(root);
  typecheck(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
