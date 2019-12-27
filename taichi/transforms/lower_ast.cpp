#include "../ir.h"

TLANG_NAMESPACE_BEGIN

template <typename T>
std::vector<T *> make_raw_pointer_list(
    const std::vector<std::unique_ptr<T>> &unique_pointers) {
  std::vector<T *> raw_pointers;
  for (auto &ptr : unique_pointers)
    raw_pointers.push_back(ptr.get());
  return raw_pointers;
}

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, Identifiers, and mutable local variables. Make
// AST SSA.
class LowerAST : public IRVisitor {
 private:
  Stmt *capturing_loop;

 public:
  LowerAST() {
    // TODO: change this to false
    allow_undefined_visitor = true;
    capturing_loop = nullptr;
  }

  Expr load_if_ptr(Expr expr) {
    if (expr.is<GlobalPtrStmt>()) {
      return load(expr);
    } else
      return expr;
  }

  void visit(Block *stmt_list) override {
    auto backup_block = current_block;
    current_block = stmt_list;
    auto stmts = make_raw_pointer_list(stmt_list->statements);
    for (auto &stmt : stmts) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  void visit(FrontendAllocaStmt *stmt) override {
    auto block = stmt->parent;
    auto ident = stmt->ident;
    TC_ASSERT(block->local_var_alloca.find(ident) ==
              block->local_var_alloca.end());
    auto lowered = std::make_unique<AllocaStmt>(stmt->ret_type.data_type);
    block->local_var_alloca.insert(std::make_pair(ident, lowered.get()));
    stmt->parent->replace_with(stmt, std::move(lowered));
    throw IRModified();
  }

  void visit(FrontendIfStmt *stmt) override {
    VecStatement flattened;
    stmt->condition->flatten(flattened);

    auto new_if = std::make_unique<IfStmt>(stmt->condition->stmt);

    new_if->true_mask = flattened.push_back<AllocaStmt>(DataType::i32);
    new_if->false_mask = flattened.push_back<AllocaStmt>(DataType::i32);

    flattened.push_back<LocalStoreStmt>(new_if->true_mask,
                                        stmt->condition->stmt);
    auto lnot_stmt_ptr = flattened.push_back<UnaryOpStmt>(
        UnaryOpType::logic_not, stmt->condition->stmt);
    flattened.push_back<LocalStoreStmt>(new_if->false_mask, lnot_stmt_ptr);

    if (stmt->true_statements) {
      new_if->true_statements = std::move(stmt->true_statements);
      new_if->true_statements->mask_var = new_if->true_mask;
    }
    if (stmt->false_statements) {
      new_if->false_statements = std::move(stmt->false_statements);
      new_if->false_statements->mask_var = new_if->false_mask;
    }

    flattened.push_back(std::move(new_if));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(FrontendPrintStmt *stmt) override {
    // expand rhs
    auto expr = load_if_ptr(stmt->expr);
    VecStatement flattened;
    expr->flatten(flattened);
    flattened.push_back<PrintStmt>(expr->stmt, stmt->str);
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(FrontendBreakStmt *stmt) override {
    TC_ASSERT_INFO(
        capturing_loop->is<WhileStmt>(),
        "The loop capturing 'break' must be a while loop instead of for loop.");
    auto while_stmt = capturing_loop->as<WhileStmt>();
    VecStatement stmts;
    auto const_true = stmts.push_back<ConstStmt>(TypedConstant((int32)0));
    stmts.push_back<WhileControlStmt>(while_stmt->mask, const_true);
    stmt->parent->replace_with(stmt, stmts);
    throw IRModified();
  }

  void visit(FrontendWhileStmt *stmt) override {
    // transform into a structure as
    // while (1) { cond; if (no active) break; original body...}
    auto cond = stmt->cond;
    VecStatement flattened;
    cond->flatten(flattened);
    auto cond_stmt = flattened.back().get();

    auto &&new_while = std::make_unique<WhileStmt>(std::move(stmt->body));
    auto mask = std::make_unique<AllocaStmt>(DataType::i32);
    new_while->mask = mask.get();
    auto &stmts = new_while->body;
    for (int i = 0; i < (int)flattened.size(); i++) {
      stmts->insert(std::move(flattened[i]), i);
    }
    // insert break
    stmts->insert(
        std::make_unique<WhileControlStmt>(new_while->mask, cond_stmt),
        flattened.size());
    stmt->insert_before_me(std::make_unique<AllocaStmt>(DataType::i32));
    auto &&const_stmt =
        std::make_unique<ConstStmt>(TypedConstant((int32)0xFFFFFFFF));
    auto const_stmt_ptr = const_stmt.get();
    stmt->insert_before_me(std::move(mask));
    stmt->insert_before_me(std::move(const_stmt));
    stmt->insert_before_me(
        std::make_unique<LocalStoreStmt>(new_while->mask, const_stmt_ptr));
    new_while->body->mask_var = new_while->mask;
    stmt->parent->replace_with(stmt, std::move(new_while));
    // insert an alloca for the mask
    throw IRModified();
  }

  void visit(WhileStmt *stmt) override {
    auto old_capturing_loop = capturing_loop;
    capturing_loop = stmt;
    stmt->body->accept(this);
    capturing_loop = old_capturing_loop;
  }

  void visit(FrontendForStmt *stmt) override {
    VecStatement flattened;
    // insert an alloca here
    for (int i = 0; i < (int)stmt->loop_var_id.size(); i++) {
      flattened.push_back<AllocaStmt>(DataType::i32);
      stmt->parent->local_var_alloca[stmt->loop_var_id[i]] =
          flattened.back().get();
    }

    if (stmt->is_ranged()) {
      TC_ASSERT(stmt->loop_var_id.size() == 1);
      auto begin = stmt->begin;
      auto end = stmt->end;
      begin->flatten(flattened);
      end->flatten(flattened);
      auto &&new_for = std::make_unique<RangeForStmt>(
          stmt->parent->lookup_var(stmt->loop_var_id[0]), begin->stmt,
          end->stmt, std::move(stmt->body), stmt->vectorize, stmt->parallelize,
          stmt->strictly_serialized);
      new_for->block_dim = stmt->block_dim;
      flattened.push_back(std::move(new_for));
    } else {
      std::vector<Stmt *> vars(stmt->loop_var_id.size());
      for (int i = 0; i < (int)stmt->loop_var_id.size(); i++) {
        vars[i] = stmt->parent->lookup_var(stmt->loop_var_id[i]);
      }
      auto &&new_for = std::make_unique<StructForStmt>(
          vars, stmt->global_var.cast<GlobalVariableExpression>()->snode,
          std::move(stmt->body), stmt->vectorize, stmt->parallelize);
      new_for->scratch_opt = stmt->scratch_opt;
      new_for->block_dim = stmt->block_dim;
      flattened.push_back(std::move(new_for));
    }
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(RangeForStmt *for_stmt) override {
    auto old_capturing_loop = capturing_loop;
    capturing_loop = for_stmt;
    for_stmt->body->accept(this);
    capturing_loop = old_capturing_loop;
  }

  void visit(StructForStmt *for_stmt) override {
    auto old_capturing_loop = capturing_loop;
    capturing_loop = for_stmt;
    for_stmt->body->accept(this);
    capturing_loop = old_capturing_loop;
  }

  void visit(FrontendEvalStmt *stmt) override {
    // expand rhs
    auto expr = stmt->expr;
    VecStatement flattened;
    expr->flatten(flattened);
    stmt->eval_expr.cast<EvalExpression>()->stmt_ptr = stmt->expr->stmt;
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(FrontendAssignStmt *assign) override {
    // expand rhs
    auto expr = assign->rhs;
    VecStatement flattened;
    expr->flatten(flattened);
    if (assign->lhs.is<IdExpression>()) {  // local variable
      // emit local store stmt
      flattened.push_back<LocalStoreStmt>(
          assign->parent->lookup_var(assign->lhs.cast<IdExpression>()->id),
          expr->stmt);
    } else {  // global variable
      TC_ASSERT(assign->lhs.is<GlobalPtrExpression>());
      auto global_ptr = assign->lhs.cast<GlobalPtrExpression>();
      global_ptr->flatten(flattened);
      flattened.push_back<GlobalStoreStmt>(flattened.back().get(), expr->stmt);
    }
    flattened.back()->set_tb(assign->tb);
    assign->parent->replace_with(assign, flattened);
    throw IRModified();
  }

  void visit(FrontendAtomicStmt *stmt) override {
    // replace atomic sub with negative atomic add
    if (stmt->op_type == AtomicOpType::sub) {
      stmt->val.set(Expr::make<UnaryOpExpression>(UnaryOpType::neg, stmt->val));
      stmt->op_type = AtomicOpType::add;
    }
    // expand rhs
    auto expr = stmt->val;
    VecStatement flattened;
    expr->flatten(flattened);
    if (stmt->dest.is<IdExpression>()) {  // local variable
      // emit local store stmt
      auto alloca =
          stmt->parent->lookup_var(stmt->dest.cast<IdExpression>()->id);
      flattened.push_back<AtomicOpStmt>(stmt->op_type, alloca, expr->stmt);
    } else {  // global variable
      TC_ASSERT(stmt->dest.is<GlobalPtrExpression>());
      auto global_ptr = stmt->dest.cast<GlobalPtrExpression>();
      global_ptr->flatten(flattened);
      flattened.push_back<AtomicOpStmt>(stmt->op_type, flattened.back().get(),
                                        expr->stmt);
    }
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    // expand rhs
    Stmt *val_stmt = nullptr;
    VecStatement flattened;
    if (stmt->val.expr) {
      auto expr = stmt->val;
      expr->flatten(flattened);
      val_stmt = expr->stmt;
    }
    std::vector<Stmt *> indices_stmt(stmt->indices.size(), nullptr);

    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      stmt->indices[i]->flatten(flattened);
      indices_stmt[i] = stmt->indices[i]->stmt;
    }

    auto ptr =
        flattened.push_back<GlobalPtrStmt>(stmt->snode->parent, indices_stmt);
    flattened.push_back<SNodeOpStmt>(stmt->op_type, stmt->snode->parent, ptr,
                                     val_stmt);

    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(FrontendAssertStmt *stmt) override {
    // expand rhs
    Stmt *val_stmt = nullptr;
    VecStatement flattened;
    if (stmt->val.expr) {
      auto expr = stmt->val;
      expr->flatten(flattened);
      val_stmt = expr->stmt;
    }
    flattened.push_back(Stmt::make<AssertStmt>(stmt->text, val_stmt));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  void visit(FrontendArgStoreStmt *stmt) override {
    // expand value
    VecStatement flattened;
    stmt->expr->flatten(flattened);
    flattened.push_back(
        Stmt::make<ArgStoreStmt>(stmt->arg_id, flattened.back().get()));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModified();
  }

  static void run(IRNode *node) {
    LowerAST inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void lower(IRNode *root) {
  return LowerAST::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END