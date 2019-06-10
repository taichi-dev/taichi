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
    current_block->statements.push_back(std::move(stmt));
  }

  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = primal->adjoint;
    TC_ASSERT(alloca_->is<AllocaStmt>());
    auto alloca = alloca_->as<AllocaStmt>();
    TC_ASSERT(alloca->width() == 1);
    auto local_load = Stmt::make<LocalLoadStmt>(LocalAddress(alloca, 0));
    auto add =
        Stmt::make<BinaryOpStmt>(BinaryOpType::add, local_load.get(), value);
    auto local_store = Stmt::make<LocalStoreStmt>(alloca, add.get());
    insert_back(std::move(local_load));
    insert_back(std::move(add));
    insert_back(std::move(local_store));
  }

  Stmt *alloc(Stmt *stmt) {
    if (stmt->adjoint != nullptr) {
      // create the alloca
      auto alloca = Stmt::make<AllocaStmt>(1, DataType::unknown);
      stmt->adjoint = alloca.get();
      current_block->statements.insert(current_block->statements.begin(),
                                       std::move(alloca));
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
      accumulate(bin->lhs, bin->adjoint);
      accumulate(bin->rhs, bin->adjoint);
    } else {
      TC_NOT_IMPLEMENTED
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
    TC_NOT_IMPLEMENTED
  }

  void visit(GlobalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(LocalLoadStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(LocalStoreStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(GlobalLoadStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(GlobalStoreStmt *stmt) override {
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
  typecheck(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
