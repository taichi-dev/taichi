#pragma once

#include <string>
#include <vector>

#include "taichi/lang_util.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expression.h"

TLANG_NAMESPACE_BEGIN

// Frontend Statements

class FrontendAllocaStmt : public Stmt {
 public:
  Identifier ident;

  FrontendAllocaStmt(const Identifier &lhs, DataType type) : ident(lhs) {
    ret_type = VectorType(1, type);
  }

  TI_DEFINE_ACCEPT
};

class FrontendSNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  ExprGroup indices;
  Expr val;

  FrontendSNodeOpStmt(SNodeOpType op_type,
                      SNode *snode,
                      const ExprGroup &indices,
                      const Expr &val = Expr(nullptr));

  TI_DEFINE_ACCEPT
};

class FrontendAssertStmt : public Stmt {
 public:
  std::string text;
  Expr val;

  FrontendAssertStmt(const std::string &text, const Expr &val)
      : text(text), val(val) {
  }

  TI_DEFINE_ACCEPT
};

class FrontendAssignStmt : public Stmt {
 public:
  Expr lhs, rhs;

  FrontendAssignStmt(const Expr &lhs, const Expr &rhs);

  TI_DEFINE_ACCEPT
};

class FrontendIfStmt : public Stmt {
 public:
  Expr condition;
  std::unique_ptr<Block> true_statements, false_statements;

  FrontendIfStmt(const Expr &condition) : condition(load_if_ptr(condition)) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
};

class FrontendPrintStmt : public Stmt {
 public:
  Expr expr;
  std::string str;

  FrontendPrintStmt(const Expr &expr, const std::string &str)
      : expr(load_if_ptr(expr)), str(str) {
  }

  TI_DEFINE_ACCEPT
};

class FrontendEvalStmt : public Stmt {
 public:
  Expr expr;
  Expr eval_expr;

  FrontendEvalStmt(const Expr &expr) : expr(load_if_ptr(expr)) {
  }

  TI_DEFINE_ACCEPT
};

class FrontendForStmt : public Stmt {
 public:
  Expr begin, end;
  Expr global_var;
  std::unique_ptr<Block> body;
  std::vector<Identifier> loop_var_id;
  int vectorize;
  int parallelize;
  bool strictly_serialized;
  ScratchPadOptions scratch_opt;
  int block_dim;

  bool is_ranged() const {
    if (global_var.expr == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  FrontendForStmt(const ExprGroup &loop_var, const Expr &global_var);

  FrontendForStmt(const Expr &loop_var, const Expr &begin, const Expr &end);

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
};

class FrontendFuncDefStmt : public Stmt {
 public:
  std::string funcid;
  std::unique_ptr<Block> body;

  FrontendFuncDefStmt(const std::string &funcid) : funcid(funcid) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
};

class FrontendBreakStmt : public Stmt {
 public:
  FrontendBreakStmt() {
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
};

class FrontendContinueStmt : public Stmt {
 public:
  FrontendContinueStmt() = default;

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
};

class FrontendWhileStmt : public Stmt {
 public:
  Expr cond;
  std::unique_ptr<Block> body;

  FrontendWhileStmt(const Expr &cond) : cond(load_if_ptr(cond)) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
};

class FrontendKernelReturnStmt : public Stmt {
 public:
  Expr value;

  FrontendKernelReturnStmt(const Expr &value) : value(value) {
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
};

// Expressions

class ArgLoadExpression : public Expression {
 public:
  int arg_id;

  ArgLoadExpression(int arg_id) : arg_id(arg_id) {
  }

  std::string serialize() override {
    return fmt::format("arg[{}]", arg_id);
  }

  void flatten(FlattenContext *ctx) override {
    auto ran = std::make_unique<ArgLoadStmt>(arg_id);
    ctx->push_back(std::move(ran));
    stmt = ctx->back_stmt();
  }
};

class RandExpression : public Expression {
 public:
  DataType dt;

  RandExpression(DataType dt) : dt(dt) {
  }

  std::string serialize() override {
    return fmt::format("rand<{}>()", data_type_name(dt));
  }

  void flatten(FlattenContext *ctx) override {
    auto ran = std::make_unique<RandStmt>(dt);
    ctx->push_back(std::move(ran));
    stmt = ctx->back_stmt();
  }
};

class UnaryOpExpression : public Expression {
 public:
  UnaryOpType type;
  Expr operand;
  DataType cast_type;

  UnaryOpExpression(UnaryOpType type, const Expr &operand)
      : type(type), operand(smart_load(operand)) {
    cast_type = DataType::unknown;
  }

  bool is_cast() const;

  std::string serialize() override;

  void flatten(FlattenContext *ctx) override;
};

class BinaryOpExpression : public Expression {
 public:
  BinaryOpType type;
  Expr lhs, rhs;

  BinaryOpExpression(const BinaryOpType &type, const Expr &lhs, const Expr &rhs)
      : type(type) {
    this->lhs.set(smart_load(lhs));
    this->rhs.set(smart_load(rhs));
  }

  std::string serialize() override {
    return fmt::format("({} {} {})", lhs->serialize(),
                       binary_op_type_symbol(type), rhs->serialize());
  }

  void flatten(FlattenContext *ctx) override {
    // if (stmt)
    //  return;
    lhs->flatten(ctx);
    rhs->flatten(ctx);
    ctx->push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
    ctx->stmts.back()->tb = tb;
    stmt = ctx->back_stmt();
  }
};

class TernaryOpExpression : public Expression {
 public:
  TernaryOpType type;
  Expr op1, op2, op3;

  TernaryOpExpression(TernaryOpType type,
                      const Expr &op1,
                      const Expr &op2,
                      const Expr &op3)
      : type(type) {
    this->op1.set(load_if_ptr(op1));
    this->op2.set(load_if_ptr(op2));
    this->op3.set(load_if_ptr(op3));
  }

  std::string serialize() override {
    return fmt::format("{}({} {} {})", ternary_type_name(type),
                       op1->serialize(), op2->serialize(), op3->serialize());
  }

  void flatten(FlattenContext *ctx) override {
    // if (stmt)
    //  return;
    op1->flatten(ctx);
    op2->flatten(ctx);
    op3->flatten(ctx);
    ctx->push_back(
        std::make_unique<TernaryOpStmt>(type, op1->stmt, op2->stmt, op3->stmt));
    stmt = ctx->back_stmt();
  }
};

class ExternalTensorExpression : public Expression {
 public:
  DataType dt;
  int dim;
  int arg_id;

  ExternalTensorExpression(const DataType &dt, int dim, int arg_id)
      : dt(dt), dim(dim), arg_id(arg_id) {
    set_attribute("dim", std::to_string(dim));
  }

  std::string serialize() override {
    return fmt::format("{}d_ext_arr", dim);
  }

  void flatten(FlattenContext *ctx) override {
    auto ptr = Stmt::make<ArgLoadStmt>(arg_id, true);
    ctx->push_back(std::move(ptr));
    stmt = ctx->back_stmt();
  }
};

class GlobalVariableExpression : public Expression {
 public:
  Identifier ident;
  DataType dt;
  SNode *snode;
  bool has_ambient;
  TypedConstant ambient_value;
  bool is_primal;
  Expr adjoint;

  GlobalVariableExpression(DataType dt, const Identifier &ident)
      : ident(ident), dt(dt) {
    snode = nullptr;
    has_ambient = false;
    is_primal = true;
  }

  GlobalVariableExpression(SNode *snode) : snode(snode) {
    dt = snode->dt;
    has_ambient = false;
    is_primal = true;
  }

  void set_snode(SNode *snode) {
    this->snode = snode;
    set_attribute("dim", std::to_string(snode->num_active_indices));
  }

  std::string serialize() override {
    return "#" + ident.name();
  }

  void flatten(FlattenContext *ctx) override {
    TI_ASSERT(snode->num_active_indices == 0);
    auto ptr = Stmt::make<GlobalPtrStmt>(LaneAttribute<SNode *>(snode),
                                         std::vector<Stmt *>());
    ctx->push_back(std::move(ptr));
  }
};

class GlobalPtrExpression : public Expression {
 public:
  Expr var;
  ExprGroup indices;

  GlobalPtrExpression(const Expr &var, const ExprGroup &indices)
      : var(var), indices(indices) {
  }

  std::string serialize() override;

  void flatten(FlattenContext *ctx) override;

  bool is_lvalue() const override {
    return true;
  }
};

class EvalExpression : public Expression {
 public:
  Stmt *stmt_ptr;
  int stmt_id;
  EvalExpression(Stmt *stmt) : stmt_ptr(stmt), stmt_id(stmt_ptr->id) {
    // cache stmt->id since it may be released later
  }

  std::string serialize() override {
    return fmt::format("%{}", stmt_id);
  }

  void flatten(FlattenContext *ctx) override {
    stmt = stmt_ptr;
  }
};

class RangeAssumptionExpression : public Expression {
 public:
  Expr input, base;
  int low, high;

  RangeAssumptionExpression(const Expr &input,
                            const Expr &base,
                            int low,
                            int high)
      : input(input), base(base), low(low), high(high) {
  }

  std::string serialize() override {
    return fmt::format("assume_in_range({}{:+d} <= ({}) < {}{:+d})",
                       base.serialize(), low, input.serialize(),
                       base.serialize(), high);
  }

  void flatten(FlattenContext *ctx) override {
    input->flatten(ctx);
    base->flatten(ctx);
    ctx->push_back(
        Stmt::make<RangeAssumptionStmt>(input->stmt, base->stmt, low, high));
    stmt = ctx->back_stmt();
  }
};

class IdExpression : public Expression {
 public:
  Identifier id;
  IdExpression(const std::string &name = "") : id(name) {
  }
  IdExpression(const Identifier &id) : id(id) {
  }

  std::string serialize() override {
    return id.name();
  }

  void flatten(FlattenContext *ctx) override {
    ctx->push_back(std::make_unique<LocalLoadStmt>(
        LocalAddress(ctx->current_block->lookup_var(id), 0)));
    stmt = ctx->back_stmt();
  }

  bool is_lvalue() const override {
    return true;
  }
};

// ti.atomic_*() is an expression with side effect.
class AtomicOpExpression : public Expression {
 public:
  AtomicOpType op_type;
  Expr dest, val;

  AtomicOpExpression(AtomicOpType op_type, const Expr &dest, const Expr &val)
      : op_type(op_type), dest(dest), val(val) {
  }

  std::string serialize() override;

  void flatten(FlattenContext *ctx) override;
};

class SNodeOpExpression : public Expression {
 public:
  SNode *snode;
  SNodeOpType op_type;
  ExprGroup indices;
  Expr value;

  SNodeOpExpression(SNode *snode, SNodeOpType op_type, const ExprGroup &indices)
      : snode(snode), op_type(op_type), indices(indices) {
  }

  SNodeOpExpression(SNode *snode,
                    SNodeOpType op_type,
                    const ExprGroup &indices,
                    const Expr &value)
      : snode(snode), op_type(op_type), indices(indices), value(value) {
  }

  std::string serialize() override;

  void flatten(FlattenContext *ctx) override;
};

class GlobalLoadExpression : public Expression {
 public:
  Expr ptr;
  GlobalLoadExpression(const Expr &ptr) : ptr(ptr) {
  }

  std::string serialize() override {
    return "gbl load " + ptr.serialize();
  }

  void flatten(FlattenContext *ctx) override {
    ptr->flatten(ctx);
    ctx->push_back(std::make_unique<GlobalLoadStmt>(ptr->stmt));
    stmt = ctx->back_stmt();
  }
};

class ConstExpression : public Expression {
 public:
  TypedConstant val;

  template <typename T>
  ConstExpression(const T &x) : val(x) {
  }

  std::string serialize() override {
    return val.stringify();
  }

  void flatten(FlattenContext *ctx) override {
    ctx->push_back(Stmt::make<ConstStmt>(val));
    stmt = ctx->back_stmt();
  }
};

TLANG_NAMESPACE_END
