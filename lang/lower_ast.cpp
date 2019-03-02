#include "ir.h"

TLANG_NAMESPACE_BEGIN

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, Identifiers, and mutable local variables. Make
// AST SSA.
class LowerAST : public IRVisitor {
 public:
  LowerAST() {
    allow_undefined_visitor = true;
  }

  Expr load_if_ptr(Expr expr) {
    if (expr.is<GlobalPtrStmt>()) {
      return load(expr);
    } else
      return expr;
  }

  void visit(Block *stmt_list) {
    auto backup_block = current_block;
    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  void visit(FrontendAllocaStmt *stmt) {
    auto block = stmt->parent;
    auto ident = stmt->ident;
    TC_ASSERT(block->local_var_alloca.find(ident) ==
              block->local_var_alloca.end());
    auto lowered = std::make_unique<AllocaStmt>(stmt->ret_type.data_type);
    block->local_var_alloca.insert(std::make_pair(ident, lowered.get()));
    stmt->parent->replace_with(stmt, std::move(lowered));
    throw IRModifiedException();
  }

  void visit(FrontendIfStmt *stmt) override {
    VecStatement flattened;
    stmt->condition->flatten(flattened);

    auto new_if = std::make_unique<IfStmt>(stmt->condition->stmt);

    auto true_mask = std::make_unique<AllocaStmt>(DataType::i32);
    auto false_mask = std::make_unique<AllocaStmt>(DataType::i32);
    new_if->true_mask = true_mask.get();
    new_if->false_mask = false_mask.get();

    flattened.push_back(std::move(true_mask));
    flattened.push_back(std::move(false_mask));

    auto insert = [&](std::unique_ptr<Stmt> &&stmt) {
      flattened.push_back(std::move(stmt));
    };

    insert(std::make_unique<LocalStoreStmt>(new_if->true_mask,
                                            stmt->condition->stmt));
    auto &&lnot_stmt =
        std::make_unique<UnaryOpStmt>(UnaryType::lnot, stmt->condition->stmt);
    auto lnot_stmt_ptr = lnot_stmt.get();
    insert(std::move(lnot_stmt));
    insert(std::make_unique<LocalStoreStmt>(new_if->false_mask, lnot_stmt_ptr));

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
    throw IRModifiedException();
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(FrontendPrintStmt *stmt) {
    // expand rhs
    auto expr = load_if_ptr(stmt->expr);
    VecStatement flattened;
    expr->flatten(flattened);
    auto print = std::make_unique<PrintStmt>(expr->stmt, stmt->str);
    flattened.push_back(std::move(print));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModifiedException();
  }

  void visit(FrontendWhileStmt *stmt) {
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
    stmt->insert_before(std::make_unique<AllocaStmt>(DataType::i32));
    auto &&const_stmt = std::make_unique<ConstStmt>((int)0xFFFFFFFF);
    auto const_stmt_ptr = const_stmt.get();
    stmt->insert_before(std::move(mask));
    stmt->insert_before(std::move(const_stmt));
    stmt->insert_before(
        std::make_unique<LocalStoreStmt>(new_while->mask, const_stmt_ptr));
    new_while->body->mask_var = new_while->mask;
    stmt->parent->replace_with(stmt, std::move(new_while));
    // insert an alloca for the mask
    throw IRModifiedException();
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(FrontendForStmt *stmt) {
    auto begin = stmt->begin;
    auto end = stmt->end;

    VecStatement flattened;

    // insert an alloca here
    flattened.push_back(std::make_unique<AllocaStmt>(DataType::i32));
    stmt->parent->local_var_alloca[stmt->loop_var_id] = flattened.back().get();

    begin->flatten(flattened);
    end->flatten(flattened);

    auto &&new_for = std::make_unique<RangeForStmt>(
        stmt->parent->lookup_var(stmt->loop_var_id), begin->stmt, end->stmt,
        std::move(stmt->body), stmt->vectorize, stmt->parallelize);
    new_for->body->inner_loop_variable =
        stmt->parent->lookup_var(stmt->loop_var_id);
    flattened.push_back(std::move(new_for));
    stmt->parent->replace_with(stmt, flattened);
    throw IRModifiedException();
  }

  void visit(RangeForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(FrontendAssignStmt *assign) {
    // expand rhs
    auto expr = assign->rhs;
    VecStatement flattened;
    expr->flatten(flattened);
    if (assign->lhs.is<IdExpression>()) {  // local variable
      // emit local store stmt
      auto local_store = std::make_unique<LocalStoreStmt>(
          assign->parent->lookup_var(assign->lhs.cast<IdExpression>()->id),
          expr->stmt);
      flattened.push_back(std::move(local_store));
    } else {  // global variable
      TC_ASSERT(assign->lhs.is<GlobalPtrExpression>());
      auto global_ptr = assign->lhs.cast<GlobalPtrExpression>();
      global_ptr->flatten(flattened);
      auto global_store =
          std::make_unique<GlobalStoreStmt>(flattened.back().get(), expr->stmt);
      flattened.push_back(std::move(global_store));
    }
    assign->parent->replace_with(assign, flattened);
    throw IRModifiedException();
  }

  static void run(IRNode *node) {
    LowerAST inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (IRModifiedException) {
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