#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend.h"
#include <typeinfo>

TLANG_NAMESPACE_BEGIN

// Do automatic differentiation pass in the reverse order (reverse-mode AD)

class ConvertLocalVar : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(AllocaStmt *alloc) override {
    TI_ASSERT(alloc->width() == 1);
    bool load_only = irpass::analysis::gather_statements(
                         alloc->parent,
                         [&](Stmt *s) {
                           if (auto store = s->cast<LocalStoreStmt>())
                             return store->ptr == alloc;
                           else if (auto atomic = s->cast<AtomicOpStmt>()) {
                             return atomic->dest == alloc;
                           } else {
                             return false;
                           }
                         })
                         .empty();
    if (!load_only) {
      alloc->replace_with(
          Stmt::make<StackAllocaStmt>(alloc->ret_type.data_type, 16));
      // TODO: remove 16 here
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    if (stmt->ptr[0].var->is<StackAllocaStmt>())
      stmt->replace_with(Stmt::make<StackLoadTopStmt>(stmt->ptr[0].var));
  }

  void visit(LocalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    stmt->replace_with(Stmt::make<StackPushStmt>(stmt->ptr, stmt->data));
  }
};

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

  Stmt *cmp_lt(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::cmp_lt, load(op1), load(op2));
  }

  Stmt *sub(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::sub, load(op1), load(op2));
  }

  Stmt *div(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::div, load(op1), load(op2));
  }

  Stmt *sel(Stmt *op1, Stmt *op2, Stmt *op3) {
    return insert<TernaryOpStmt>(TernaryOpType::select, load(op1), load(op2),
                                 load(op3));
  }

  Stmt *cos(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::cos, load(op1));
  }

  Stmt *sin(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::sin, load(op1));
  }

  Stmt *log(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::log, load(op1));
  }

  Stmt *pow(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::pow, load(op1), load(op2));
  }

 public:
  Block *current_block;
  Block *alloca_block;
  std::map<Stmt *, Stmt *> adjoint_stmt;
  int for_depth;

  MakeAdjoint() {
    current_block = nullptr;
    alloca_block = nullptr;
    // Note:
    // MakeAdjoint acts on the block with if's and without struct/range/while
    // loops. This is basically a straight-line code with forking and merging
    // due to if's. Therefore we the adjoint allocas must belong to this block
    // for it to be visible. We call this block `alloca_block`.

    for_depth = 0;
  }

  static void run(IRNode *node) {
    auto p = MakeAdjoint();
    node->accept(&p);
  }

  void visit(Block *block) override {
    if (current_block == nullptr) {
      // serial
      alloca_block = block;
    }
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

  // Accumulate [value] to the adjoint of [primal]
  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = adjoint(primal);
    if (!alloca_ || alloca_->is<ConstStmt>())
      return;  // primal may be int variable
    if (alloca_->is<StackAllocaStmt>()) {
      auto alloca = alloca_->cast<StackAllocaStmt>();
      if (needs_grad(alloca->ret_type.data_type)) {
        insert<StackAccAdjointStmt>(alloca, load(value));
      }
    } else {
      TI_ASSERT(alloca_->is<AllocaStmt>());
      auto alloca = alloca_->as<AllocaStmt>();
      TI_ASSERT(alloca->width() == 1);
      auto local_load = insert<LocalLoadStmt>(LocalAddress(alloca, 0));
      insert<LocalStoreStmt>(alloca, add(local_load, value));
    }
  }

  Stmt *adjoint(Stmt *stmt) {
    if (!needs_grad(stmt->ret_type.data_type)) {
      return constant(0);
    }
    if (adjoint_stmt.find(stmt) == adjoint_stmt.end()) {
      // normal SSA cases

      // create the alloca
      // auto alloca =
      //    Stmt::make<AllocaStmt>(1, get_current_program().config.gradient_dt);
      // maybe it's better to use the statement data type than the default type
      auto alloca = Stmt::make<AllocaStmt>(1, stmt->ret_type.data_type);
      adjoint_stmt[stmt] = alloca.get();
      alloca_block->insert(std::move(alloca), 0);
    }
    return adjoint_stmt[stmt];
  }

  void visit(AllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(StackAllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(ArgLoadStmt *stmt) override {
    // do nothing.
  }

  void visit(LoopIndexStmt *stmt) override {
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
      TI_NOT_IMPLEMENTED
    } else if (stmt->op_type == UnaryOpType::tanh) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt), sub(constant(1), sqr(stmt))));
    } else if (stmt->op_type == UnaryOpType::asin) {
      accumulate(
          stmt->operand,
          mul(adjoint(stmt),
              div(constant(1), sqrt(sub(constant(1), sqr(stmt->operand))))));
    } else if (stmt->op_type == UnaryOpType::acos) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt),
                     negate(div(constant(1),
                                sqrt(sub(constant(1), sqr(stmt->operand)))))));
    } else if (stmt->op_type == UnaryOpType::exp) {
      accumulate(stmt->operand, mul(adjoint(stmt), stmt));
    } else if (stmt->op_type == UnaryOpType::log) {
      accumulate(stmt->operand, div(adjoint(stmt), stmt->operand));
    } else if (stmt->op_type == UnaryOpType::sqrt) {
      accumulate(stmt->operand,
                 mul(adjoint(stmt), div(constant(0.5f), sqrt(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      if (is_real(stmt->cast_type) &&
          is_real(stmt->operand->ret_type.data_type)) {
        accumulate(stmt->operand, adjoint(stmt));
      }
    } else if (stmt->op_type == UnaryOpType::logic_not) {
      // do nothing
    } else {
      TI_P(unary_op_type_name(stmt->op_type));
      TI_NOT_IMPLEMENTED
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
      // d (x * y) = y * dx + x * dy
      accumulate(bin->lhs, mul(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, mul(adjoint(bin), bin->lhs));
    } else if (bin->op_type == BinaryOpType::mod) {
      // Do nothing
    } else if (bin->op_type == BinaryOpType::div) {
      accumulate(bin->lhs, div(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs),
                                      mul(bin->rhs, bin->rhs))));
    } else if (bin->op_type == BinaryOpType::atan2) {
      auto numerator = add(sqr(bin->lhs), sqr(bin->rhs));
      accumulate(bin->lhs, div(mul(adjoint(bin), bin->rhs), numerator));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs), numerator)));
    } else if (bin->op_type == BinaryOpType::pow) {
      // d (x ^ y) = x ^ (y-1) * (y * dx + log(x) * x * dy)
      auto common_coeff =
          pow(bin->lhs, sub(bin->rhs, constant(1)));  // x ^ (y-1)
      accumulate(bin->lhs, mul(adjoint(bin), mul(bin->rhs, common_coeff)));
      accumulate(bin->rhs, mul(adjoint(bin), mul(log(bin->lhs),
                                                 mul(bin->lhs, common_coeff))));
    } else if (bin->op_type == BinaryOpType::min ||
               bin->op_type == BinaryOpType::max) {
      auto cmp = bin->op_type == BinaryOpType::min ? cmp_lt(bin->lhs, bin->rhs)
                                                   : cmp_lt(bin->rhs, bin->lhs);
      auto zero = insert<ConstStmt>(TypedConstant(bin->ret_type.data_type));
      accumulate(bin->lhs, sel(cmp, adjoint(bin), zero));
      accumulate(bin->rhs, sel(cmp, zero, adjoint(bin)));
    } else if (bin->op_type == BinaryOpType::floordiv) {
      // do nothing
    } else if (is_comparison(bin->op_type) || is_bit_op(bin->op_type)) {
      // do nothing
    } else {
      TI_WARN("gradient of binary op {}", binary_op_type_name(bin->op_type));
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    TI_ASSERT(stmt->op_type == TernaryOpType::select);
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
      new_if->true_statements = std::make_unique<Block>();
      auto old_current_block = current_block;

      current_block = new_if->true_statements.get();
      for (int i = if_stmt->true_statements->statements.size() - 1; i >= 0;
           i--) {
        if_stmt->true_statements->statements[i]->accept(this);
      }

      current_block = old_current_block;
    }
    if (if_stmt->false_statements) {
      new_if->false_statements = std::make_unique<Block>();
      auto old_current_block = current_block;
      current_block = new_if->false_statements.get();
      for (int i = if_stmt->false_statements->statements.size() - 1; i >= 0;
           i--) {
        if_stmt->false_statements->statements[i]->accept(this);
      }
      current_block = old_current_block;
    }
    insert_back(std::move(new_if));
  }

  void visit(PrintStmt *print_stmt) override {
    // do nothing
  }

  void visit(ConstStmt *const_stmt) override {
    // do nothing
  }

  void visit(WhileControlStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void visit(ContinueStmt *stmt) override {
    TI_NOT_IMPLEMENTED;
  }

  void visit(WhileStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void visit(RangeForStmt *for_stmt) override {
    if (for_depth > 0)  // reverse non-parallelized for-loops
      for_stmt->reverse();
    for_depth += 1;
    alloca_block = for_stmt->body.get();
    for_stmt->body->accept(this);
    for_depth -= 1;
  }

  void visit(StructForStmt *for_stmt) override {
    for_depth += 1;
    alloca_block = for_stmt->body.get();
    for_stmt->body->accept(this);
    for_depth -= 1;
  }

  void visit(GlobalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(LocalLoadStmt *stmt) override {
    // TI_ASSERT(!needs_grad(stmt->ret_type.data_type));
  }

  void visit(StackLoadTopStmt *stmt) override {
    if (needs_grad(stmt->ret_type.data_type))
      insert<StackAccAdjointStmt>(stmt->stack, load(adjoint(stmt)));
  }

  void visit(StackPushStmt *stmt) override {
    accumulate(stmt->v, insert<StackLoadTopAdjStmt>(stmt->stack));
    insert<StackPopStmt>(stmt->stack);
  }

  Stmt *load(Stmt *alloc) {
    TI_ASSERT(alloc != nullptr);
    if (alloc->is<AllocaStmt>()) {
      return insert<LocalLoadStmt>(LocalAddress(alloc, 0));
    } else {
      // non alloca
      return alloc;
    }
  }

  bool gradients_stopped(GlobalLoadStmt *stmt, SNode *snode) {
    for (auto block = stmt->parent; block; block = block->parent) {
      for (auto s : block->stop_gradients) {
        if (s == snode) {
          return true;
        }
      }
    }
    return false;
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to adjoint
    GlobalPtrStmt *ptr = stmt->ptr->as<GlobalPtrStmt>();
    TI_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    if (!snodes[0]->has_grad()) {
      // No adjoint SNode. Do nothing
      return;
    }
    if (gradients_stopped(stmt, snodes[0])) {
      // gradients stopped, do nothing.
      return;
    }
    TI_ASSERT(snodes[0]->get_grad() != nullptr);
    snodes[0] = snodes[0]->get_grad();
    auto adj_ptr = insert<GlobalPtrStmt>(snodes, ptr->indices);
    insert<AtomicOpStmt>(AtomicOpType::add, adj_ptr, load(adjoint(stmt)));
  }

  void visit(GlobalStoreStmt *stmt) override {
    // erase and replace with global load adjoint
    GlobalPtrStmt *ptr = stmt->ptr->as<GlobalPtrStmt>();
    TI_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    if (!snodes[0]->has_grad()) {
      // no gradient (likely integer types)
      return;
    }
    TI_ASSERT(snodes[0]->get_grad() != nullptr);
    snodes[0] = snodes[0]->get_grad();
    auto adjoint_ptr = insert<GlobalPtrStmt>(snodes, ptr->indices);
    auto load = insert<GlobalLoadStmt>(adjoint_ptr);
    accumulate(stmt->data, load);
    stmt->parent->erase(stmt);
  }

  void visit(AtomicOpStmt *stmt) override {
    // erase and replace with global load adjoint
    GlobalPtrStmt *ptr = stmt->dest->as<GlobalPtrStmt>();
    TI_ASSERT(ptr->width() == 1);
    auto snodes = ptr->snodes;
    if (snodes[0]->has_grad()) {
      TI_ASSERT(snodes[0]->get_grad() != nullptr);
      snodes[0] = snodes[0]->get_grad();
      auto adjoint_ptr = insert<GlobalPtrStmt>(snodes, ptr->indices);
      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
    } else {
      // no gradient (likely integer types)
    }
    stmt->parent->erase(stmt);
  }

  void visit(ElementShuffleStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void visit(AssertStmt *stmt) override {
    // do nothing
  }

  void visit(RangeAssumptionStmt *stmt) override {
    // do nothing
  }

  void visit(LinearizeStmt *stmt) override {
    // do nothing
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    // do nothing
  }

  void visit(IntegerOffsetStmt *stmt) override {
    // do nothing
  }
};

class BackupSSA : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  Block *current_block;
  std::map<Stmt *, Stmt *> backup_alloca;

  BackupSSA() {
    current_block = nullptr;
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  Stmt *load(Stmt *stmt) {
    if (backup_alloca.find(stmt) == backup_alloca.end()) {
      auto alloca =
          Stmt::make<AllocaStmt>(stmt->width(), stmt->ret_type.data_type);
      auto alloca_ptr = alloca.get();
      TI_ASSERT(current_block != nullptr);
      current_block->insert(std::move(alloca), 0);
      stmt->insert_after_me(Stmt::make<LocalStoreStmt>(alloca_ptr, stmt));
      backup_alloca[stmt] = alloca_ptr;
    }
    return backup_alloca[stmt];
  }

  void visit(Stmt *stmt) override {
    std::vector<Block *> leaf_to_root;
    auto t = stmt->parent;
    while (t != nullptr) {
      leaf_to_root.push_back(t);
      t = t->parent;
    }
    int num_operands = stmt->get_operands().size();
    for (int i = 0; i < num_operands; i++) {
      auto op = stmt->operand(i);
      if (std::find(leaf_to_root.begin(), leaf_to_root.end(), op->parent) ==
              leaf_to_root.end() &&
          !op->is<AllocaStmt>()) {
        auto alloca = load(op);
        TI_ASSERT(op->width() == 1);
        stmt->set_operand(i, stmt->insert_before_me(Stmt::make<LocalLoadStmt>(
                                 LocalAddress(alloca, 0))));
      }
    }
  }

  void visit(RangeForStmt *stmt) override {
    auto old_current_block = current_block;
    current_block = stmt->body.get();
    stmt->body->accept(this);
    current_block = old_current_block;
  }

  void visit(StructForStmt *stmt) override {
    auto old_current_block = current_block;
    current_block = stmt->body.get();
    stmt->body->accept(this);
    current_block = old_current_block;
  }

  void visit(WhileStmt *stmt) override {
    TI_ERROR("WhileStmt not supported by autodiff for now");
  }

  void visit(Block *block) override {
    // top-level block case
    auto old_current_block = current_block;
    if (old_current_block == nullptr)
      current_block = block;
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      TI_ASSERT(!stmt->erased);
      stmt->accept(this);
    }
    if (old_current_block == nullptr)
      current_block = old_current_block;
  }
};

namespace irpass {

void make_adjoint(IRNode *root, bool use_stack) {
  if (use_stack) {
    fix_block_parents(root);
    ConvertLocalVar converter;
    root->accept(&converter);
    typecheck(root);
    MakeAdjoint::run(root);
    typecheck(root);
    fix_block_parents(root);
    BackupSSA b;
    root->accept(&b);
    typecheck(root);
  } else {
    MakeAdjoint::run(root);
    typecheck(root);
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
