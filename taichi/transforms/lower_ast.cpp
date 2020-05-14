#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

namespace {

using FlattenContext = Expression::FlattenContext;
using StructForOffsets = std::unordered_map<StructForStmt *, std::vector<int>>;

template <typename T>
std::vector<T *> make_raw_pointer_list(
    const std::vector<std::unique_ptr<T>> &unique_pointers) {
  std::vector<T *> raw_pointers;
  for (auto &ptr : unique_pointers)
    raw_pointers.push_back(ptr.get());
  return raw_pointers;
}

}  // namespace

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, Identifiers, and mutable local variables. Make
// AST SSA.
class LowerAST : public IRVisitor {
 private:
  Stmt *capturing_loop;
  std::unordered_set<Stmt *> detected_fors_with_break;
  Block *current_block;

  StructForOffsets struct_for_offsets;

  FlattenContext make_flatten_ctx() {
    FlattenContext fctx;
    fctx.current_block = this->current_block;
    return fctx;
  }

 public:
  explicit LowerAST(const std::unordered_set<Stmt *> &_detected_fors_with_break)
      : detected_fors_with_break(_detected_fors_with_break),
        current_block(nullptr) {
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
    auto backup_block = this->current_block;
    this->current_block = stmt_list;
    auto stmts = make_raw_pointer_list(stmt_list->statements);
    for (auto &stmt : stmts) {
      stmt->accept(this);
    }
    this->current_block = backup_block;
  }

  void visit(FrontendAllocaStmt *stmt) override {
    auto block = stmt->parent;
    auto ident = stmt->ident;
    TI_ASSERT(block->local_var_alloca.find(ident) ==
              block->local_var_alloca.end());
    auto lowered = std::make_unique<AllocaStmt>(stmt->ret_type.data_type);
    block->local_var_alloca.insert(std::make_pair(ident, lowered.get()));
    stmt->parent->replace_with(stmt, std::move(lowered));
    throw IRModified();
  }

  void visit(FrontendIfStmt *stmt) override {
    auto fctx = make_flatten_ctx();
    stmt->condition->flatten(&fctx);

    auto new_if = std::make_unique<IfStmt>(stmt->condition->stmt);

    new_if->true_mask = fctx.push_back<AllocaStmt>(DataType::i32);
    new_if->false_mask = fctx.push_back<AllocaStmt>(DataType::i32);

    fctx.push_back<LocalStoreStmt>(new_if->true_mask, stmt->condition->stmt);
    auto lnot_stmt_ptr = fctx.push_back<UnaryOpStmt>(UnaryOpType::logic_not,
                                                     stmt->condition->stmt);
    fctx.push_back<LocalStoreStmt>(new_if->false_mask, lnot_stmt_ptr);

    if (stmt->true_statements) {
      new_if->true_statements = std::move(stmt->true_statements);
      new_if->true_statements->mask_var = new_if->true_mask;
    }
    if (stmt->false_statements) {
      new_if->false_statements = std::move(stmt->false_statements);
      new_if->false_statements->mask_var = new_if->false_mask;
    }

    fctx.push_back(std::move(new_if));
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
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
    auto fctx = make_flatten_ctx();
    expr->flatten(&fctx);
    fctx.push_back<PrintStmt>(expr->stmt, stmt->str);
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    throw IRModified();
  }

  void visit(FrontendBreakStmt *stmt) override {
    auto while_stmt = capturing_loop->as<WhileStmt>();
    VecStatement stmts;
    auto const_true = stmts.push_back<ConstStmt>(TypedConstant((int32)0));
    stmts.push_back<WhileControlStmt>(while_stmt->mask, const_true);
    stmt->parent->replace_with(stmt, std::move(stmts));
    throw IRModified();
  }

  void visit(FrontendContinueStmt *stmt) override {
    stmt->parent->replace_with(stmt, Stmt::make<ContinueStmt>());
  }

  void visit(FrontendWhileStmt *stmt) override {
    // transform into a structure as
    // while (1) { cond; if (no active) break; original body...}
    auto cond = stmt->cond;
    auto fctx = make_flatten_ctx();
    cond->flatten(&fctx);
    auto cond_stmt = fctx.back_stmt();

    auto &&new_while = std::make_unique<WhileStmt>(std::move(stmt->body));
    auto mask = std::make_unique<AllocaStmt>(DataType::i32);
    new_while->mask = mask.get();
    auto &stmts = new_while->body;
    stmts->insert(std::move(fctx.stmts), /*location=*/0);
    // insert break
    stmts->insert(
        std::make_unique<WhileControlStmt>(new_while->mask, cond_stmt),
        fctx.stmts.size());
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
    auto fctx = make_flatten_ctx();
    // insert an alloca here
    for (int i = 0; i < (int)stmt->loop_var_id.size(); i++) {
      fctx.push_back<AllocaStmt>(DataType::i32);
      stmt->parent->local_var_alloca[stmt->loop_var_id[i]] = fctx.back_stmt();
    }

    if (stmt->is_ranged()) {
      TI_ASSERT(stmt->loop_var_id.size() == 1);
      auto begin = stmt->begin;
      auto end = stmt->end;
      begin->flatten(&fctx);
      end->flatten(&fctx);
      bool is_good_range_for =
          capturing_loop == nullptr ||
          detected_fors_with_break.find(stmt) == detected_fors_with_break.end();
      // #578: a good range for is a range for that doesn't contains a break
      // statement
      if (is_good_range_for) {
        auto &&new_for = std::make_unique<RangeForStmt>(
            stmt->parent->lookup_var(stmt->loop_var_id[0]), begin->stmt,
            end->stmt, std::move(stmt->body), stmt->vectorize,
            stmt->parallelize, stmt->block_dim, stmt->strictly_serialized);
        fctx.push_back(std::move(new_for));
      } else {
        // transform into a structure as
        // i = begin; while (1) { if (i >= end) break; original body; i += 1; }
        auto loop_var = stmt->parent->lookup_var(stmt->loop_var_id[0]);
        fctx.push_back<LocalStoreStmt>(loop_var, begin->stmt);
        auto loop_var_addr = LaneAttribute<LocalAddress>(
            LocalAddress(loop_var->as<AllocaStmt>(), 0));
        VecStatement load_and_compare;
        auto loop_var_load_stmt =
            load_and_compare.push_back<LocalLoadStmt>(loop_var_addr);
        auto cond_stmt = load_and_compare.push_back<BinaryOpStmt>(
            BinaryOpType::cmp_lt, loop_var_load_stmt, end->stmt);

        auto &&new_while = std::make_unique<WhileStmt>(std::move(stmt->body));
        auto mask = std::make_unique<AllocaStmt>(DataType::i32);
        new_while->mask = mask.get();
        auto &stmts = new_while->body;
        for (int i = 0; i < (int)load_and_compare.size(); i++) {
          stmts->insert(std::move(load_and_compare[i]), i);
        }

        VecStatement increase_and_store;
        auto const_one =
            increase_and_store.push_back<ConstStmt>(TypedConstant((int32)1));
        auto loop_var_add_one = increase_and_store.push_back<BinaryOpStmt>(
            BinaryOpType::add, loop_var_load_stmt, const_one);
        increase_and_store.push_back<LocalStoreStmt>(loop_var,
                                                     loop_var_add_one);
        for (int i = 0; i < (int)increase_and_store.size(); i++) {
          stmts->insert(std::move(increase_and_store[i]), stmts->size());
        }
        // insert break
        stmts->insert(
            std::make_unique<WhileControlStmt>(new_while->mask, cond_stmt),
            load_and_compare.size());

        stmt->insert_before_me(std::make_unique<AllocaStmt>(DataType::i32));
        auto &&const_stmt =
            std::make_unique<ConstStmt>(TypedConstant((int32)0xFFFFFFFF));
        auto const_stmt_ptr = const_stmt.get();
        stmt->insert_before_me(std::move(mask));
        stmt->insert_before_me(std::move(const_stmt));
        stmt->insert_before_me(
            std::make_unique<LocalStoreStmt>(new_while->mask, const_stmt_ptr));
        new_while->body->mask_var = new_while->mask;
        fctx.push_back(std::move(new_while));
        stmt->parent->replace_with(stmt, std::move(fctx.stmts));
        throw IRModified();
      }
    } else {
      std::vector<Stmt *> vars(stmt->loop_var_id.size());
      for (int i = 0; i < (int)stmt->loop_var_id.size(); i++) {
        vars[i] = stmt->parent->lookup_var(stmt->loop_var_id[i]);
      }
      auto snode = stmt->global_var.cast<GlobalVariableExpression>()->snode;
      std::vector<int> offsets;
      if (snode->type == SNodeType::place) {
        /* Note:
         * for i in x:
         *   x[i] = 0
         *
         * has the same effect as
         *
         * for i in x.parent():
         *   x[i] = 0
         *
         * (unless x has index offsets)*/
        offsets = snode->index_offsets;
        snode = snode->parent;
      }
      auto &&new_for = std::make_unique<StructForStmt>(
          vars, snode, std::move(stmt->body), stmt->vectorize,
          stmt->parallelize, stmt->block_dim);
      new_for->scratch_opt = stmt->scratch_opt;
      struct_for_offsets[new_for.get()] = offsets;
      fctx.push_back(std::move(new_for));
    }
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
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

  void visit(FrontendKernelReturnStmt *stmt) override {
    auto expr = stmt->value;
    auto fctx = make_flatten_ctx();
    expr->flatten(&fctx);
    fctx.push_back<KernelReturnStmt>(fctx.back_stmt());
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    throw IRModified();
  }

  void visit(FrontendEvalStmt *stmt) override {
    // expand rhs
    auto expr = stmt->expr;
    auto fctx = make_flatten_ctx();
    expr->flatten(&fctx);
    stmt->eval_expr.cast<EvalExpression>()->stmt_ptr = stmt->expr->stmt;
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    throw IRModified();
  }

  void visit(FrontendAssignStmt *assign) override {
    // expand rhs
    auto expr = assign->rhs;
    auto fctx = make_flatten_ctx();
    expr->flatten(&fctx);
    if (assign->lhs.is<IdExpression>()) {  // local variable
      // emit local store stmt
      fctx.push_back<LocalStoreStmt>(
          assign->parent->lookup_var(assign->lhs.cast<IdExpression>()->id),
          expr->stmt);
    } else {  // global variable
      TI_ASSERT(assign->lhs.is<GlobalPtrExpression>());
      auto global_ptr = assign->lhs.cast<GlobalPtrExpression>();
      global_ptr->flatten(&fctx);
      fctx.push_back<GlobalStoreStmt>(fctx.back_stmt(), expr->stmt);
    }
    fctx.stmts.back()->set_tb(assign->tb);
    assign->parent->replace_with(assign, std::move(fctx.stmts));
    throw IRModified();
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    // expand rhs
    Stmt *val_stmt = nullptr;
    auto fctx = make_flatten_ctx();
    if (stmt->val.expr) {
      auto expr = stmt->val;
      expr->flatten(&fctx);
      val_stmt = expr->stmt;
    }
    std::vector<Stmt *> indices_stmt(stmt->indices.size(), nullptr);

    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      stmt->indices[i]->flatten(&fctx);
      indices_stmt[i] = stmt->indices[i]->stmt;
    }

    if (stmt->snode->type == SNodeType::dynamic) {
      auto ptr = fctx.push_back<GlobalPtrStmt>(stmt->snode, indices_stmt);
      fctx.push_back<SNodeOpStmt>(stmt->op_type, stmt->snode, ptr, val_stmt);
    } else if (stmt->snode->type == SNodeType::pointer ||
               stmt->snode->type == SNodeType::hash ||
               stmt->snode->type == SNodeType::dynamic ||
               stmt->snode->type == SNodeType::bitmasked) {
      TI_ASSERT(SNodeOpStmt::activation_related(stmt->op_type));
      fctx.push_back<SNodeOpStmt>(stmt->op_type, stmt->snode, indices_stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }

    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    throw IRModified();
  }

  void visit(FrontendAssertStmt *stmt) override {
    // expand rhs
    Stmt *val_stmt = nullptr;
    auto fctx = make_flatten_ctx();
    if (stmt->val.expr) {
      auto expr = stmt->val;
      expr->flatten(&fctx);
      val_stmt = expr->stmt;
    }
    fctx.push_back<AssertStmt>(stmt->text, val_stmt);
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    throw IRModified();
  }

  static StructForOffsets run(IRNode *node) {
    LowerAST inst(irpass::analysis::detect_fors_with_break(node));
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
    return inst.struct_for_offsets;
  }
};

class FixStructForOffsets : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  StructForOffsets offsets;

  void visit(StructForStmt *stmt) {
    if (const auto &it = offsets.find(stmt);
        it != offsets.end() && !it->second.empty()) {
      auto offset = it->second;
      std::vector<Stmt *> loop_vars_with_offset(stmt->loop_vars.size());
      VecStatement new_statements;
      for (int i = 0; i < (int)stmt->loop_vars.size(); i++) {
        auto old_alloca = stmt->loop_vars[i];

        auto new_alloca =
            new_statements.push_back<AllocaStmt>(1, DataType::i32);

        auto load = new_statements.push_back<LocalLoadStmt>(
            LocalAddress(old_alloca, 0));

        auto offset_const =
            new_statements.push_back<ConstStmt>(TypedConstant(offset[i]));

        auto add = new_statements.push_back<BinaryOpStmt>(BinaryOpType::add,
                                                          load, offset_const);

        new_statements.push_back<LocalStoreStmt>(new_alloca, add);

        loop_vars_with_offset[i] = new_alloca;

        irpass::replace_all_usages_with(stmt->body.get(), old_alloca,
                                        new_alloca);
      }

      stmt->body->insert(std::move(new_statements), 0);
    }
  }

  static void run(IRNode *root, const StructForOffsets &offsets) {
    FixStructForOffsets inst;
    inst.offsets = offsets;
    root->accept(&inst);
  }
};

namespace irpass {

void lower(IRNode *root) {
  auto offsets = LowerAST::run(root);
  FixStructForOffsets::run(root, offsets);
  convert_into_loop_index(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
