#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/system/profiler.h"

#include <unordered_set>

TLANG_NAMESPACE_BEGIN

namespace {

using FlattenContext = Expression::FlattenContext;

template <typename Vec>
std::vector<typename Vec::value_type::pointer> make_raw_pointer_list(
    const Vec &unique_pointers) {
  std::vector<typename Vec::value_type::pointer> raw_pointers;
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
  Stmt *capturing_loop_;
  std::unordered_set<Stmt *> detected_fors_with_break_;
  Block *current_block_;
  int current_block_depth_;

  FlattenContext make_flatten_ctx() {
    FlattenContext fctx;
    fctx.current_block = this->current_block_;
    return fctx;
  }

 public:
  explicit LowerAST(const std::unordered_set<Stmt *> &_detected_fors_with_break)
      : detected_fors_with_break_(_detected_fors_with_break),
        current_block_(nullptr),
        current_block_depth_(0) {
    // TODO: change this to false
    allow_undefined_visitor = true;
    capturing_loop_ = nullptr;
  }

  void visit(Block *stmt_list) override {
    auto backup_block = this->current_block_;
    this->current_block_ = stmt_list;
    auto stmts = make_raw_pointer_list(stmt_list->statements);
    current_block_depth_++;
    for (auto &stmt : stmts) {
      stmt->accept(this);
    }
    current_block_depth_--;
    this->current_block_ = backup_block;
  }

  void visit(FrontendAllocaStmt *stmt) override {
    auto block = stmt->parent;
    auto ident = stmt->ident;
    TI_ASSERT(block->local_var_to_stmt.find(ident) ==
              block->local_var_to_stmt.end());
    if (stmt->ret_type->is<TensorType>()) {
      auto tensor_type = stmt->ret_type->cast<TensorType>();
      auto lowered = std::make_unique<AllocaStmt>(
          tensor_type->get_shape(), tensor_type->get_element_type(),
          stmt->is_shared);
      block->local_var_to_stmt.insert(std::make_pair(ident, lowered.get()));
      stmt->parent->replace_with(stmt, std::move(lowered));
    } else {
      auto lowered = std::make_unique<AllocaStmt>(stmt->ret_type);
      block->local_var_to_stmt.insert(std::make_pair(ident, lowered.get()));
      stmt->parent->replace_with(stmt, std::move(lowered));
    }
  }

  void visit(FrontendIfStmt *stmt) override {
    auto fctx = make_flatten_ctx();
    flatten_rvalue(stmt->condition, &fctx);

    auto new_if = std::make_unique<IfStmt>(stmt->condition->stmt);

    if (stmt->true_statements) {
      new_if->set_true_statements(std::move(stmt->true_statements));
    }
    if (stmt->false_statements) {
      new_if->set_false_statements(std::move(stmt->false_statements));
    }
    auto pif = new_if.get();
    fctx.push_back(std::move(new_if));
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    pif->accept(this);
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
    std::vector<Stmt *> stmts;
    std::vector<std::variant<Stmt *, std::string>> new_contents;
    auto fctx = make_flatten_ctx();
    for (auto c : stmt->contents) {
      if (std::holds_alternative<Expr>(c)) {
        auto x = std::get<Expr>(c);
        flatten_rvalue(x, &fctx);
        stmts.push_back(x->stmt);
        new_contents.push_back(x->stmt);
      } else {
        auto x = std::get<std::string>(c);
        new_contents.push_back(x);
      }
    }
    fctx.push_back<PrintStmt>(new_contents);
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
  }

  void visit(FrontendBreakStmt *stmt) override {
    auto while_stmt = capturing_loop_->as<WhileStmt>();
    VecStatement stmts;
    auto const_true = stmts.push_back<ConstStmt>(TypedConstant((int32)0));
    stmts.push_back<WhileControlStmt>(while_stmt->mask, const_true);
    stmt->parent->replace_with(stmt, std::move(stmts));
  }

  void visit(FrontendContinueStmt *stmt) override {
    stmt->parent->replace_with(stmt, Stmt::make<ContinueStmt>());
  }

  void visit(FrontendWhileStmt *stmt) override {
    // transform into a structure as
    // while (1) { cond; if (no active) break; original body...}
    auto cond = stmt->cond;
    auto fctx = make_flatten_ctx();
    flatten_rvalue(cond, &fctx);
    auto cond_stmt = fctx.back_stmt();

    auto &&new_while = std::make_unique<WhileStmt>(std::move(stmt->body));
    auto mask = std::make_unique<AllocaStmt>(PrimitiveType::i32);
    new_while->mask = mask.get();
    auto &stmts = new_while->body;
    stmts->insert(std::move(fctx.stmts), /*location=*/0);
    // insert break
    stmts->insert(
        std::make_unique<WhileControlStmt>(new_while->mask, cond_stmt),
        fctx.stmts.size());
    auto &&const_stmt =
        std::make_unique<ConstStmt>(TypedConstant((int32)0xFFFFFFFF));
    auto const_stmt_ptr = const_stmt.get();
    stmt->insert_before_me(std::move(mask));
    stmt->insert_before_me(std::move(const_stmt));
    stmt->insert_before_me(
        std::make_unique<LocalStoreStmt>(new_while->mask, const_stmt_ptr));
    auto pwhile = new_while.get();
    stmt->parent->replace_with(stmt, std::move(new_while));
    pwhile->accept(this);
    // insert an alloca for the mask
  }

  void visit(WhileStmt *stmt) override {
    auto old_capturing_loop = capturing_loop_;
    capturing_loop_ = stmt;
    stmt->body->accept(this);
    capturing_loop_ = old_capturing_loop;
  }

  void visit(LoopIndexStmt *stmt) override {
    // do nothing
  }

  void visit(BinaryOpStmt *stmt) override {
    // do nothing
  }

  void visit(FrontendForStmt *stmt) override {
    auto fctx = make_flatten_ctx();
    if (stmt->is_ranged()) {
      TI_ASSERT(stmt->loop_var_id.size() == 1);
      auto begin = stmt->begin;
      auto end = stmt->end;
      flatten_rvalue(begin, &fctx);
      flatten_rvalue(end, &fctx);
      bool is_good_range_for = detected_fors_with_break_.find(stmt) ==
                               detected_fors_with_break_.end();
      // #578: a good range for is a range for that doesn't contains a break
      // statement
      if (is_good_range_for) {
        auto &&new_for = std::make_unique<RangeForStmt>(
            begin->stmt, end->stmt, std::move(stmt->body),
            stmt->is_bit_vectorized, stmt->num_cpu_threads, stmt->block_dim,
            stmt->strictly_serialized);
        new_for->body->insert(std::make_unique<LoopIndexStmt>(new_for.get(), 0),
                              0);
        new_for->body->local_var_to_stmt[stmt->loop_var_id[0]] =
            new_for->body->statements[0].get();
        fctx.push_back(std::move(new_for));
      } else {
        // transform into a structure as
        // i = begin - 1; while (1) { i += 1; if (i >= end) break; original
        // body; }
        fctx.push_back<AllocaStmt>(PrimitiveType::i32);
        auto loop_var = fctx.back_stmt();
        stmt->parent->local_var_to_stmt[stmt->loop_var_id[0]] = loop_var;
        auto const_one = fctx.push_back<ConstStmt>(TypedConstant((int32)1));
        auto begin_minus_one = fctx.push_back<BinaryOpStmt>(
            BinaryOpType::sub, begin->stmt, const_one);
        fctx.push_back<LocalStoreStmt>(loop_var, begin_minus_one);
        auto loop_var_addr = loop_var->as<AllocaStmt>();
        VecStatement load_and_compare;
        auto loop_var_load_stmt =
            load_and_compare.push_back<LocalLoadStmt>(loop_var_addr);
        auto loop_var_add_one = load_and_compare.push_back<BinaryOpStmt>(
            BinaryOpType::add, loop_var_load_stmt, const_one);

        auto cond_stmt = load_and_compare.push_back<BinaryOpStmt>(
            BinaryOpType::cmp_lt, loop_var_add_one, end->stmt);

        auto &&new_while = std::make_unique<WhileStmt>(std::move(stmt->body));
        auto mask = std::make_unique<AllocaStmt>(PrimitiveType::i32);
        new_while->mask = mask.get();

        // insert break
        load_and_compare.push_back<WhileControlStmt>(new_while->mask,
                                                     cond_stmt);
        load_and_compare.push_back<LocalStoreStmt>(loop_var, loop_var_add_one);
        auto &stmts = new_while->body;
        for (int i = 0; i < (int)load_and_compare.size(); i++) {
          stmts->insert(std::move(load_and_compare[i]), i);
        }

        stmt->insert_before_me(
            std::make_unique<AllocaStmt>(PrimitiveType::i32));
        auto &&const_stmt =
            std::make_unique<ConstStmt>(TypedConstant((int32)0xFFFFFFFF));
        auto const_stmt_ptr = const_stmt.get();
        stmt->insert_before_me(std::move(mask));
        stmt->insert_before_me(std::move(const_stmt));
        stmt->insert_before_me(
            std::make_unique<LocalStoreStmt>(new_while->mask, const_stmt_ptr));
        fctx.push_back(std::move(new_while));
      }
    } else if (stmt->mesh_for) {
      auto &&new_for = std::make_unique<MeshForStmt>(
          stmt->mesh, stmt->element_type, std::move(stmt->body),
          stmt->is_bit_vectorized, stmt->num_cpu_threads, stmt->block_dim);
      new_for->body->insert(std::make_unique<LoopIndexStmt>(new_for.get(), 0),
                            0);
      new_for->body->local_var_to_stmt[stmt->loop_var_id[0]] =
          new_for->body->statements[0].get();
      new_for->mem_access_opt = stmt->mem_access_opt;
      new_for->fields_registered = true;
      fctx.push_back(std::move(new_for));
    } else if (stmt->global_var.is<GlobalVariableExpression>()) {
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

      // Climb up one more level if inside bit_struct.
      // Note that when looping over bit_structs, we generate
      // struct for on their parent node instead of itself for
      // higher performance.
      if (snode->type == SNodeType::bit_struct)
        snode = snode->parent;

      auto &&new_for = std::make_unique<StructForStmt>(
          snode, std::move(stmt->body), stmt->is_bit_vectorized,
          stmt->num_cpu_threads, stmt->block_dim);
      new_for->index_offsets = offsets;
      VecStatement new_statements;
      for (int i = 0; i < (int)stmt->loop_var_id.size(); i++) {
        Stmt *loop_index = new_statements.push_back<LoopIndexStmt>(
            new_for.get(), snode->physical_index_position[i]);
        if ((int)offsets.size() > i && offsets[i] != 0) {
          auto offset_const =
              new_statements.push_back<ConstStmt>(TypedConstant(offsets[i]));
          auto result = new_statements.push_back<BinaryOpStmt>(
              BinaryOpType::add, loop_index, offset_const);
          loop_index = result;
        }
        new_for->body->local_var_to_stmt[stmt->loop_var_id[i]] = loop_index;
      }
      new_for->body->insert(std::move(new_statements), 0);
      new_for->mem_access_opt = stmt->mem_access_opt;
      fctx.push_back(std::move(new_for));
    } else {
      auto tensor = stmt->global_var.cast<ExternalTensorExpression>();
      std::vector<Stmt *> shape;
      for (int i = 0; i < tensor->dim - abs(tensor->element_dim); i++) {
        shape.push_back(fctx.push_back<ExternalTensorShapeAlongAxisStmt>(
            i, tensor->arg_id));
      }
      Stmt *begin = fctx.push_back<ConstStmt>(TypedConstant(0));
      Stmt *end = fctx.push_back<ConstStmt>(TypedConstant(1));
      for (int i = 0; i < (int)shape.size(); i++) {
        end = fctx.push_back<BinaryOpStmt>(BinaryOpType::mul, end, shape[i]);
      }
      // TODO: add a note explaining why shape might be empty.
      auto &&new_for = std::make_unique<RangeForStmt>(
          begin, end, std::move(stmt->body), stmt->is_bit_vectorized,
          stmt->num_cpu_threads, stmt->block_dim, stmt->strictly_serialized,
          /*range_hint=*/fmt::format("arg {}", tensor->arg_id));
      VecStatement new_statements;
      Stmt *loop_index =
          new_statements.push_back<LoopIndexStmt>(new_for.get(), 0);
      for (int i = (int)shape.size() - 1; i >= 0; i--) {
        Stmt *loop_var = new_statements.push_back<BinaryOpStmt>(
            BinaryOpType::mod, loop_index, shape[i]);
        new_for->body->local_var_to_stmt[stmt->loop_var_id[i]] = loop_var;
        std::vector<uint32_t> decoration = {
            uint32_t(DecorationStmt::Decoration::kLoopUnique), uint32_t(i)};
        new_statements.push_back<DecorationStmt>(loop_var, decoration);
        loop_index = new_statements.push_back<BinaryOpStmt>(
            BinaryOpType::div, loop_index, shape[i]);
      }
      new_for->body->insert(std::move(new_statements), 0);
      fctx.push_back(std::move(new_for));
    }
    auto pfor = fctx.stmts.back().get();
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
    pfor->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    auto old_capturing_loop = capturing_loop_;
    capturing_loop_ = for_stmt;
    for_stmt->body->accept(this);
    capturing_loop_ = old_capturing_loop;
  }

  void visit(StructForStmt *for_stmt) override {
    auto old_capturing_loop = capturing_loop_;
    capturing_loop_ = for_stmt;
    for_stmt->body->accept(this);
    capturing_loop_ = old_capturing_loop;
  }

  void visit(MeshForStmt *for_stmt) override {
    auto old_capturing_loop = capturing_loop_;
    capturing_loop_ = for_stmt;
    for_stmt->body->accept(this);
    capturing_loop_ = old_capturing_loop;
  }

  void visit(FrontendReturnStmt *stmt) override {
    auto expr_group = stmt->values;
    auto fctx = make_flatten_ctx();
    std::vector<Stmt *> return_ele;
    for (auto &x : expr_group.exprs) {
      flatten_rvalue(x, &fctx);
      return_ele.push_back(fctx.back_stmt());
    }
    fctx.push_back<ReturnStmt>(return_ele);
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
  }

  void visit(FrontendAssignStmt *assign) override {
    auto dest = assign->lhs;
    auto expr = assign->rhs;
    auto fctx = make_flatten_ctx();
    flatten_rvalue(expr, &fctx);
    if (dest.is<IdExpression>()) {
      fctx.push_back<LocalStoreStmt>(
          assign->parent->lookup_var(assign->lhs.cast<IdExpression>()->id),
          expr->stmt);
    } else if (dest.is<IndexExpression>()) {
      auto ix = dest.cast<IndexExpression>();
      flatten_lvalue(dest, &fctx);
      if (ix->is_local()) {
        fctx.push_back<LocalStoreStmt>(dest->stmt, expr->stmt);
      } else {
        fctx.push_back<GlobalStoreStmt>(dest->stmt, expr->stmt);
      }
    } else if (dest.is<StrideExpression>()) {
      flatten_lvalue(dest, &fctx);
      fctx.push_back<GlobalStoreStmt>(dest->stmt, expr->stmt);
    } else {
      TI_ASSERT(dest.is<ArgLoadExpression>() &&
                dest.cast<ArgLoadExpression>()->is_ptr);
      flatten_lvalue(dest, &fctx);
      fctx.push_back<GlobalStoreStmt>(dest->stmt, expr->stmt);
    }
    fctx.stmts.back()->set_tb(assign->tb);
    assign->parent->replace_with(assign, std::move(fctx.stmts));
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    // expand rhs
    Stmt *val_stmt = nullptr;
    auto fctx = make_flatten_ctx();
    if (stmt->val.expr) {
      auto expr = stmt->val;
      flatten_rvalue(expr, &fctx);
      val_stmt = expr->stmt;
    }
    std::vector<Stmt *> indices_stmt(stmt->indices.size(), nullptr);

    for (int i = 0; i < (int)stmt->indices.size(); i++) {
      flatten_rvalue(stmt->indices[i], &fctx);
      indices_stmt[i] = stmt->indices[i]->stmt;
    }

    if (stmt->snode->type == SNodeType::dynamic) {
      auto ptr = fctx.push_back<GlobalPtrStmt>(stmt->snode, indices_stmt);
      fctx.push_back<SNodeOpStmt>(stmt->op_type, stmt->snode, ptr, val_stmt);
    } else if (stmt->snode->type == SNodeType::pointer ||
               stmt->snode->type == SNodeType::hash ||
               stmt->snode->type == SNodeType::dense ||
               stmt->snode->type == SNodeType::bitmasked) {
      TI_ASSERT(SNodeOpStmt::activation_related(stmt->op_type));
      auto ptr = fctx.push_back<GlobalPtrStmt>(stmt->snode, indices_stmt);
      fctx.push_back<SNodeOpStmt>(stmt->op_type, stmt->snode, ptr, val_stmt);
    } else {
      TI_ERROR("The {} operation is not supported on {} SNode",
               snode_op_type_name(stmt->op_type),
               snode_type_name(stmt->snode->type));
      TI_NOT_IMPLEMENTED
    }

    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
  }

  void visit(FrontendAssertStmt *stmt) override {
    // expand rhs
    Stmt *val_stmt = nullptr;
    auto fctx = make_flatten_ctx();
    if (stmt->cond.expr) {
      auto expr = stmt->cond;
      flatten_rvalue(expr, &fctx);
      val_stmt = expr->stmt;
    }

    auto &fargs = stmt->args;  // frontend stmt args
    std::vector<Stmt *> args_stmts(fargs.size());
    for (int i = 0; i < (int)fargs.size(); ++i) {
      flatten_rvalue(fargs[i], &fctx);
      args_stmts[i] = fargs[i]->stmt;
    }
    fctx.push_back<AssertStmt>(val_stmt, stmt->text, args_stmts);
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
  }

  void visit(FrontendExprStmt *stmt) override {
    auto fctx = make_flatten_ctx();
    flatten_rvalue(stmt->val, &fctx);
    stmt->parent->replace_with(stmt, std::move(fctx.stmts));
  }

  void visit(FrontendExternalFuncStmt *stmt) override {
    auto ctx = make_flatten_ctx();
    TI_ASSERT((int)(stmt->so_func != nullptr) +
                  (int)(!stmt->asm_source.empty()) +
                  (int)(!stmt->bc_filename.empty()) ==
              1)
    std::vector<Stmt *> arg_statements, output_statements;
    if (stmt->so_func != nullptr || !stmt->asm_source.empty()) {
      for (auto &s : stmt->args) {
        flatten_rvalue(s, &ctx);
        arg_statements.push_back(s->stmt);
      }
      for (auto &s : stmt->outputs) {
        flatten_lvalue(s, &ctx);
        output_statements.push_back(s->stmt);
      }
      ctx.push_back(std::make_unique<ExternalFuncCallStmt>(
          (stmt->so_func != nullptr) ? ExternalFuncCallStmt::SHARED_OBJECT
                                     : ExternalFuncCallStmt::ASSEMBLY,
          stmt->so_func, stmt->asm_source, "", "", arg_statements,
          output_statements));
    } else {
      for (auto &s : stmt->args) {
        TI_ASSERT_INFO(
            s.is<IdExpression>(),
            "external func call via bitcode must pass in local variables.")
        flatten_lvalue(s, &ctx);
        arg_statements.push_back(s->stmt);
      }
      ctx.push_back(std::make_unique<ExternalFuncCallStmt>(
          ExternalFuncCallStmt::BITCODE, nullptr, "", stmt->bc_filename,
          stmt->bc_funcname, arg_statements, output_statements));
    }
    stmt->parent->replace_with(stmt, std::move(ctx.stmts));
  }

  static void run(IRNode *node) {
    LowerAST inst(irpass::analysis::detect_fors_with_break(node));
    node->accept(&inst);
  }
};

namespace irpass {

void lower_ast(IRNode *root) {
  TI_AUTO_PROF;
  LowerAST::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
