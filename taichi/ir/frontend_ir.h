#pragma once

#include <string>
#include <vector>

#include "taichi/ir/snode_types.h"
#include "taichi/ir/stmt_op_types.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expression.h"
#include "taichi/program/function.h"

TLANG_NAMESPACE_BEGIN

// Frontend Statements

class FrontendExprStmt : public Stmt {
 public:
  Expr val;

  FrontendExprStmt(const Expr &val) : val(val) {
  }

  TI_DEFINE_ACCEPT
};

class FrontendAllocaStmt : public Stmt {
 public:
  Identifier ident;

  FrontendAllocaStmt(const Identifier &lhs, DataType type) : ident(lhs) {
    ret_type = TypeFactory::create_vector_or_scalar_type(1, type);
  }

  FrontendAllocaStmt(const Identifier &lhs,
                     std::vector<int> shape,
                     DataType element)
      : ident(lhs) {
    ret_type = DataType(TypeFactory::create_tensor_type(shape, element));
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
  Expr cond;
  std::vector<Expr> args;

  FrontendAssertStmt(const Expr &cond, const std::string &text)
      : text(text), cond(cond) {
  }

  FrontendAssertStmt(const Expr &cond,
                     const std::string &text,
                     const std::vector<Expr> &args_)
      : text(text), cond(cond) {
    for (auto &a : args_) {
      args.push_back(load_if_ptr(a));
    }
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
  using EntryType = std::variant<Expr, std::string>;
  std::vector<EntryType> contents;

  FrontendPrintStmt(const std::vector<EntryType> &contents_) {
    for (const auto &c : contents_) {
      if (std::holds_alternative<Expr>(c))
        contents.push_back(load_if_ptr(std::get<Expr>(c)));
      else
        contents.push_back(c);
    }
  }

  TI_DEFINE_ACCEPT
};

// This statement evaluates the expression.
// The expression should have side effects otherwise the expression will do
// nothing.
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
  int bit_vectorize;
  int num_cpu_threads;
  bool strictly_serialized;
  MemoryAccessOptions mem_access_opt;
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

class FrontendReturnStmt : public Stmt {
 public:
  Expr value;

  FrontendReturnStmt(const Expr &value) : value(value) {
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
  DataType dt;

  ArgLoadExpression(int arg_id, DataType dt) : arg_id(arg_id), dt(dt) {
  }

  std::string serialize() override {
    return fmt::format("arg[{}] (dt={})", arg_id, data_type_name(dt));
  }

  void flatten(FlattenContext *ctx) override;
};

class RandExpression : public Expression {
 public:
  DataType dt;

  RandExpression(DataType dt) : dt(dt) {
  }

  std::string serialize() override {
    return fmt::format("rand<{}>()", data_type_name(dt));
  }

  void flatten(FlattenContext *ctx) override;
};

class UnaryOpExpression : public Expression {
 public:
  UnaryOpType type;
  Expr operand;
  DataType cast_type;

  UnaryOpExpression(UnaryOpType type, const Expr &operand)
      : type(type), operand(smart_load(operand)) {
    cast_type = PrimitiveType::unknown;
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

  void flatten(FlattenContext *ctx) override;
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

  void flatten(FlattenContext *ctx) override;
};

class InternalFuncCallExpression : public Expression {
 public:
  std::string func_name;
  std::vector<Expr> args;

  InternalFuncCallExpression(const std::string &func_name,
                             const std::vector<Expr> &args_)
      : func_name(func_name) {
    for (auto &a : args_) {
      args.push_back(load_if_ptr(a));
    }
  }

  std::string serialize() override {
    std::string args_str;
    for (int i = 0; i < args.size(); i++) {
      if (i != 0) {
        args_str += ", ";
      }
      args_str += args[i]->serialize();
    }
    return fmt::format("internal call {}({})", func_name, args_str);
  }

  void flatten(FlattenContext *ctx) override;
};

class ExternalFuncCallExpression : public Expression {
 public:
  void *func;
  std::string source;
  std::vector<Expr> args;
  std::vector<Expr> outputs;

  ExternalFuncCallExpression(void *func,
                             std::string const &source,
                             const std::vector<Expr> &args,
                             const std::vector<Expr> &outputs)
      : func(func), source(source), args(args), outputs(outputs) {
  }

  std::string serialize() override {
    std::string io = "inputs=";

    for (auto &s : args) {
      io += s.serialize();
    }

    io += ", outputs=";

    for (auto &s : outputs) {
      io += s.serialize();
    }

    if (func) {
      return fmt::format("call {:x} ({})", (uint64)func, io);
    } else {
      return fmt::format("asm \"{}\" ({})", source, io);
    }
  }

  void flatten(FlattenContext *ctx) override;
};

class ExternalTensorExpression : public Expression {
 public:
  DataType dt;
  int dim;
  int arg_id;
  int element_dim;  // 0: scalar; 1: vector (SOA); 2: matrix (SOA); -1: vector
                    // (AOS); -2: matrix (AOS)

  ExternalTensorExpression(const DataType &dt,
                           int dim,
                           int arg_id,
                           int element_dim)
      : dt(dt), dim(dim), arg_id(arg_id), element_dim(element_dim) {
    set_attribute("dim", std::to_string(dim));
  }

  std::string serialize() override {
    return fmt::format("{}d_ext_arr", dim);
  }

  void flatten(FlattenContext *ctx) override;
};

class GlobalVariableExpression : public Expression {
 public:
  Identifier ident;
  DataType dt;
  std::string name;
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

  void flatten(FlattenContext *ctx) override;
};

class GlobalPtrExpression : public Expression {
 public:
  SNode *snode{nullptr};
  Expr var;
  ExprGroup indices;

  GlobalPtrExpression(const Expr &var, const ExprGroup &indices)
      : var(var), indices(indices) {
  }

  GlobalPtrExpression(SNode *snode, const ExprGroup &indices)
      : snode(snode), indices(indices) {
  }

  std::string serialize() override;

  void flatten(FlattenContext *ctx) override;

  bool is_lvalue() const override {
    return true;
  }
};

class TensorElementExpression : public Expression {
 public:
  Expr var;
  ExprGroup indices;
  std::vector<int> shape;
  int layout_stride{1};

  TensorElementExpression(const Expr &var,
                          const ExprGroup &indices,
                          const std::vector<int> &shape,
                          int layout_stride)
      : var(var), indices(indices), shape(shape), layout_stride(layout_stride) {
  }

  bool is_local_tensor() const;

  bool is_global_tensor() const;

  std::string serialize() override {
    std::string s = fmt::format("{}[", var.serialize());
    for (int i = 0; i < (int)indices.size(); i++) {
      s += indices.exprs[i]->serialize();
      if (i + 1 < (int)indices.size())
        s += ", ";
    }
    s += "] (";
    for (int i = 0; i < (int)shape.size(); i++) {
      s += std::to_string(shape[i]);
      if (i + 1 < (int)shape.size())
        s += ", ";
    }
    s += ", layout_stride = " + std::to_string(layout_stride);
    s += ")";
    return s;
  }

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

  void flatten(FlattenContext *ctx) override;
};

class LoopUniqueExpression : public Expression {
 public:
  Expr input;
  std::vector<SNode *> covers;

  LoopUniqueExpression(const Expr &input, const std::vector<SNode *> &covers)
      : input(input), covers(covers) {
  }

  std::string serialize() override;

  void flatten(FlattenContext *ctx) override;
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

  void flatten(FlattenContext *ctx) override;

  Stmt *flatten_noload(FlattenContext *ctx) {
    return ctx->current_block->lookup_var(id);
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

class LocalLoadExpression : public Expression {
 public:
  Expr ptr;
  LocalLoadExpression(const Expr &ptr) : ptr(ptr) {
  }

  std::string serialize() override {
    return "lcl load " + ptr.serialize();
  }

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

  void flatten(FlattenContext *ctx) override;
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

  void flatten(FlattenContext *ctx) override;
};

class ExternalTensorShapeAlongAxisExpression : public Expression {
 public:
  Expr ptr;
  int axis;

  std::string serialize() override {
    return fmt::format("external_tensor_shape_along_axis({}, {})",
                       ptr->serialize(), axis);
  }

  ExternalTensorShapeAlongAxisExpression(const Expr &ptr, int axis)
      : ptr(ptr), axis(axis) {
  }

  void flatten(FlattenContext *ctx) override;
};

class FuncCallExpression : public Expression {
 public:
  Function *func;
  ExprGroup args;

  std::string serialize() override;

  FuncCallExpression(Function *func, const ExprGroup &args)
      : func(func), args(args) {
  }

  void flatten(FlattenContext *ctx) override;
};

class ASTBuilder {
 private:
  std::vector<Block *> stack;

 public:
  ASTBuilder(Block *initial) {
    stack.push_back(initial);
  }

  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

  struct ScopeGuard {
    ASTBuilder *builder;
    Block *list;
    ScopeGuard(ASTBuilder *builder, Block *list)
        : builder(builder), list(list) {
      builder->stack.push_back(list);
    }

    ~ScopeGuard() {
      builder->stack.pop_back();
    }
  };

  std::unique_ptr<ScopeGuard> create_scope(std::unique_ptr<Block> &list);
  Block *current_block();
  Stmt *get_last_stmt();
  void stop_gradient(SNode *);
};

ASTBuilder &current_ast_builder();

class FrontendContext {
 private:
  std::unique_ptr<ASTBuilder> current_builder;
  std::unique_ptr<Block> root_node;

 public:
  FrontendContext();

  ASTBuilder &builder() {
    return *current_builder;
  }

  IRNode *root();

  std::unique_ptr<Block> get_root() {
    return std::move(root_node);
  }
};

TLANG_NAMESPACE_END
