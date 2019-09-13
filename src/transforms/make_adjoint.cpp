#include <typeinfo>
#include "../ir.h"
#include <taichi/lang.h>

TLANG_NAMESPACE_BEGIN

class MakeAdjoint : public IRVisitor {
 private:
  Stmt *constant(float32 x) {
    return insert<ConstStmt>(TypedConstant(x));
  }

  // utils
  Stmt *sgn(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::sgn, load(inp));
  }

  // utils
  Stmt *negate(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::neg, load(inp));
  }

  Stmt *sqrt(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::sqrt, load(inp));
  }

  Stmt *mul(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::mul, load(op1), load(op2));
  }

  Stmt *sqr(Stmt *op1) {
    return mul(op1, op1);
  }

  Stmt *add(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::add, load(op1), load(op2));
  }

  Stmt *sub(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::sub, load(op1), load(op2));
  }

  Stmt *div(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::div, load(op1), load(op2));
  }

  Stmt *cos(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::cos, load(op1));
  }

  Stmt *sin(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::sin, load(op1));
  }

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

  Stmt *insert_back(std::unique_ptr<Stmt> &&stmt) {
    auto ptr = stmt.get();
    current_block->insert(std::move(stmt), -1);
    return ptr;
  }

  template <typename T, typename... Args>
  Stmt *insert(Args &&... args) {
    return insert_back(Stmt::make<T>(args...));
  }

  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = adjoint(primal);
    if (!alloca_ || alloca_->is<ConstStmt>())
      return;  // primal may be int variable
    TC_ASSERT(alloca_->is<AllocaStmt>());
    auto alloca = alloca_->as<AllocaStmt>();
    TC_ASSERT(alloca->width() == 1);
    auto local_load = insert<LocalLoadStmt>(LocalAddress(alloca, 0));
    insert<LocalStoreStmt>(alloca, add(local_load, value));
  }

  Stmt *adjoint(Stmt *stmt) {
    if (!needs_grad(stmt->ret_type.data_type)) {
      return constant(0);
    }
    if (stmt->adjoint == nullptr) {
      // create the alloca
      // auto alloca =
      //    Stmt::make<AllocaStmt>(1, get_current_program().config.gradient_dt);
      // maybe it's better to use the statement data type than the default type
      auto alloca = Stmt::make<AllocaStmt>(1, stmt->ret_type.data_type);
      stmt->adjoint = alloca.get();
      current_block->insert(std::move(alloca), 0);
    }
    return stmt->adjoint;
  }

  void visit(AllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(ArgLoadStmt *stmt) override {
    // do nothing.
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type == UnaryOpType::floor) {
      // do nothing
    } else if (stmt->op_type == UnaryOpType::neg) {
      accumulate(stmt->operand, negate(adjoint(stmt)));
    } else if (stmt->op_type == UnaryOpType::abs) {
      accumulate(stmt->operand, mul(adjoint(stmt), sgn(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::sin) {
      accumulate(stmt->operand, mul(adjoint(stmt), cos(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cos) {
      accumulate(stmt->operand, negate(mul(adjoint(stmt), sin(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::tan) {
      TC_NOT_IMPLEMENTED
    } else if (stmt->op_type == UnaryOpType::tanh) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt), sub(constant(1), sqr(stmt))));
    } else if (stmt->op_type == UnaryOpType::exp) {
      accumulate(stmt->operand, mul(adjoint(stmt), stmt));
    } else if (stmt->op_type == UnaryOpType::log) {
      accumulate(stmt->operand, div(adjoint(stmt), stmt->operand));
    } else if (stmt->op_type == UnaryOpType::sqrt) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt), div(constant(0.5f), sqrt(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::cast) {
      if (stmt->cast_by_value && is_real(stmt->cast_type)) {
        accumulate(stmt->operand, stmt);
      }
    } else if (stmt->op_type == UnaryOpType::logic_not) {
      // do nothing
    } else {
      TC_P(unary_op_type_name(stmt->op_type));
      TC_NOT_IMPLEMENTED
    }
  }

  void visit(BinaryOpStmt *bin) override {
    if (bin->op_type == BinaryOpType::add) {
      accumulate(bin->lhs, adjoint(bin));
      accumulate(bin->rhs, adjoint(bin));
    } else if (bin->op_type == BinaryOpType::sub) {
      accumulate(bin->lhs, adjoint(bin));
      accumulate(bin->rhs, negate(adjoint(bin)));
    } else if (bin->op_type == BinaryOpType::mul) {
      accumulate(bin->lhs, mul(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, mul(adjoint(bin), bin->lhs));
    } else if (bin->op_type == BinaryOpType::div) {
      accumulate(bin->lhs, div(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs),
                                      mul(bin->rhs, bin->rhs))));
    } else if (is_comparison(bin->op_type) || is_bit_op(bin->op_type)) {
      // do nothing
    } else {
      TC_WARN("{}", binary_op_type_name(bin->op_type));
      TC_NOT_IMPLEMENTED
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    TC_ASSERT(stmt->op_type == TernaryOpType::select);
    auto zero = insert<ConstStmt>(TypedConstant(stmt->ret_type.data_type));
    accumulate(stmt->op2,
               insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1,
                                     load(adjoint(stmt)), zero));
    accumulate(stmt->op3,
               insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1, zero,
                                     load(adjoint(stmt))));
  }

  void visit(IfStmt *if_stmt) override {
    auto new_if = Stmt::make_typed<IfStmt>(if_stmt->cond);
    if (if_stmt->true_statements) {
      // make sure only global store/atomic exists
      new_if->true_statements = std::make_unique<Block>();
      auto old_current_block = current_block;

      current_block = new_if->true_statements.get();
      for (int i = 0; i < if_stmt->true_statements->statements.size(); i++) {
        // TC_ASSERT(if_stmt->true_statements[i])
        if_stmt->true_statements->statements[i]->accept(this);
      }

      current_block = old_current_block;
    }
    insert_back(std::move(new_if));
  }

  void visit(PrintStmt *print_stmt) override {
    // do nothing
    return;
  }

  void visit(ConstStmt *const_stmt) override {
    // do nothing
  }

  void visit(WhileControlStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(WhileStmt *stmt) override {
    TC_NOT_IMPLEMENTED
  }

  void visit(RangeForStmt *for_stmt) override {
    TC_WARN("Range for order not yet reversed.");
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(GlobalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(LocalLoadStmt *stmt) override {
    // do nothing
    TC_WARN("needs impl when loading something other than loop var");
  }

  void visit(LocalStoreStmt *stmt) override{TC_NOT_IMPLEMENTED}

  Stmt *load(Stmt *alloc) {
    TC_ASSERT(alloc != nullptr);
    if (alloc->is<AllocaStmt>()) {
      return insert<LocalLoadStmt>(LocalAddress(alloc, 0));
    } else {
      // non alloca
      return alloc;
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to adjoint
    GlobalPtrStmt *ptr = stmt->ptr->as<GlobalPtrStmt>();
    TC_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    if (!snodes[0]->has_grad()) {
      // No adjoint SNode. Do nothing
      return;
    }
    TC_ASSERT(snodes[0]->get_grad() != nullptr);
    snodes[0] = snodes[0]->get_grad();
    auto adj_ptr = insert<GlobalPtrStmt>(snodes, ptr->indices);
    insert<AtomicOpStmt>(AtomicOpType::add, adj_ptr, load(adjoint(stmt)));
  }

  void visit(GlobalStoreStmt *stmt) override {
    // erase and replace with global load adjoint
    GlobalPtrStmt *ptr = stmt->ptr->as<GlobalPtrStmt>();
    TC_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    TC_ASSERT(snodes[0]->get_grad() != nullptr);
    snodes[0] = snodes[0]->get_grad();
    auto adjoint_ptr = insert<GlobalPtrStmt>(snodes, ptr->indices);
    accumulate(stmt->data, insert<GlobalLoadStmt>(adjoint_ptr));
    stmt->parent->erase(stmt);
  }

  void visit(AtomicOpStmt *stmt) override {
    // erase and replace with global load adjoint
    GlobalPtrStmt *ptr = stmt->dest->as<GlobalPtrStmt>();
    TC_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    TC_ASSERT(snodes[0]->get_grad() != nullptr);
    snodes[0] = snodes[0]->get_grad();
    auto adjoint_ptr = insert<GlobalPtrStmt>(snodes, ptr->indices);
    accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
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
  // print(root);
  typecheck(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
