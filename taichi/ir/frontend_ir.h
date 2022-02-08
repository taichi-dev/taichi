#pragma once

#include <string>
#include <vector>

#include "taichi/ir/snode_types.h"
#include "taichi/ir/stmt_op_types.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expression.h"
#include "taichi/program/arch.h"
#include "taichi/program/function.h"
#include "taichi/ir/mesh.h"

TLANG_NAMESPACE_BEGIN

// Frontend Statements
class FrontendExternalFuncStmt : public Stmt {
 public:
  void *so_func;
  std::string asm_source;
  std::string bc_filename;
  std::string bc_funcname;
  std::vector<Expr> args;
  std::vector<Expr> outputs;

  FrontendExternalFuncStmt(void *so_func,
                           const std::string &asm_source,
                           const std::string &bc_filename,
                           const std::string &bc_funcname,
                           const std::vector<Expr> &args,
                           const std::vector<Expr> &outputs)
      : so_func(so_func),
        asm_source(asm_source),
        bc_filename(bc_filename),
        bc_funcname(bc_funcname),
        args(args),
        outputs(outputs) {
  }

  TI_DEFINE_ACCEPT
};

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
      args.push_back(a);
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

  FrontendIfStmt(const Expr &condition) : condition(condition) {
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
        contents.push_back(std::get<Expr>(c));
      else
        contents.push_back(c);
    }
  }

  TI_DEFINE_ACCEPT
};

class FrontendForStmt : public Stmt {
 public:
  Expr begin, end;
  Expr global_var;
  std::unique_ptr<Block> body;
  std::vector<Identifier> loop_var_id;
  int bit_vectorize;
  int num_cpu_threads;
  bool strictly_serialized;
  MemoryAccessOptions mem_access_opt;
  int block_dim;

  bool mesh_for = false;
  mesh::Mesh *mesh;
  mesh::MeshElementType element_type;

  bool is_ranged() const {
    if (global_var.expr == nullptr && !mesh_for) {
      return true;
    } else {
      return false;
    }
  }

  FrontendForStmt(const ExprGroup &loop_var, const Expr &global_var, Arch arch);

  FrontendForStmt(const ExprGroup &loop_var,
                  const mesh::MeshPtr &mesh,
                  const mesh::MeshElementType &element_type,
                  Arch arch);

  FrontendForStmt(const Expr &loop_var,
                  const Expr &begin,
                  const Expr &end,
                  Arch arch);

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

  FrontendWhileStmt(const Expr &cond) : cond(cond) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
};

class FrontendReturnStmt : public Stmt {
 public:
  ExprGroup values;

  FrontendReturnStmt(const ExprGroup &group) : values(group) {
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

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << fmt::format("arg[{}] (dt={})", arg_id, data_type_name(dt));
  }

  void flatten(FlattenContext *ctx) override;
};

class RandExpression : public Expression {
 public:
  DataType dt;

  RandExpression(DataType dt) : dt(dt) {
  }

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << fmt::format("rand<{}>()", data_type_name(dt));
  }

  void flatten(FlattenContext *ctx) override;
};

class UnaryOpExpression : public Expression {
 public:
  UnaryOpType type;
  Expr operand;
  DataType cast_type;

  UnaryOpExpression(UnaryOpType type, const Expr &operand)
      : type(type), operand(operand) {
    cast_type = PrimitiveType::unknown;
  }

  UnaryOpExpression(UnaryOpType type, const Expr &operand, DataType cast_type)
      : type(type), operand(operand), cast_type(cast_type) {
  }

  void type_check() override;

  bool is_cast() const;

  void serialize(std::ostream &ss) override;

  void flatten(FlattenContext *ctx) override;
};

class BinaryOpExpression : public Expression {
 public:
  BinaryOpType type;
  Expr lhs, rhs;

  BinaryOpExpression(const BinaryOpType &type, const Expr &lhs, const Expr &rhs)
      : type(type), lhs(lhs), rhs(rhs) {
  }

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << '(';
    lhs->serialize(ss);
    ss << ' ';
    ss << binary_op_type_symbol(type);
    ss << ' ';
    rhs->serialize(ss);
    ss << ')';
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
    this->op1.set(op1);
    this->op2.set(op2);
    this->op3.set(op3);
  }

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << ternary_type_name(type) << '(';
    op1->serialize(ss);
    ss << ' ';
    op2->serialize(ss);
    ss << ' ';
    op3->serialize(ss);
    ss << ')';
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
      args.push_back(a);
    }
  }

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << "internal call " << func_name << '(';
    std::string args_str;
    for (int i = 0; i < args.size(); i++) {
      if (i != 0) {
        ss << ", ";
      }
      args[i]->serialize(ss);
    }
    ss << ')';
  }

  void flatten(FlattenContext *ctx) override;
};

// TODO: Make this a non-expr
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

  void type_check() override {
  }

  void serialize(std::ostream &ss) override {
    ss << fmt::format("{}d_ext_arr", dim);
  }

  void flatten(FlattenContext *ctx) override;
};

// TODO: Make this a non-expr
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

  void type_check() override {
  }

  void set_snode(SNode *snode) {
    this->snode = snode;
    set_attribute("dim", std::to_string(snode->num_active_indices));
  }

  void serialize(std::ostream &ss) override {
    ss << "#" << ident.name();
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

  void type_check() override;

  void serialize(std::ostream &ss) override;

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
  int stride{0};

  TensorElementExpression(const Expr &var,
                          const ExprGroup &indices,
                          const std::vector<int> &shape,
                          int stride)
      : var(var), indices(indices), shape(shape), stride(stride) {
    // TODO: shape & indices check
  }

  void type_check() override;

  bool is_local_tensor() const;

  bool is_global_tensor() const;

  void serialize(std::ostream &ss) override {
    var.serialize(ss);
    ss << '[';
    for (int i = 0; i < (int)indices.size(); i++) {
      indices.exprs[i]->serialize(ss);
      if (i + 1 < (int)indices.size())
        ss << ", ";
    }
    ss << "] (";
    for (int i = 0; i < (int)shape.size(); i++) {
      ss << std::to_string(shape[i]);
      if (i + 1 < (int)shape.size())
        ss << ", ";
    }
    ss << ", stride = " + std::to_string(stride);
    ss << ')';
  }

  void flatten(FlattenContext *ctx) override;

  bool is_lvalue() const override {
    return true;
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

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << "assume_in_range({";
    base.serialize(ss);
    ss << fmt::format("{:+d}", low);
    ss << " <= (";
    input.serialize(ss);
    ss << ")  < ";
    base.serialize(ss);
    ss << fmt::format("{:+d})", high);
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

  void type_check() override;

  void serialize(std::ostream &ss) override;

  void flatten(FlattenContext *ctx) override;
};

class IdExpression : public Expression {
 public:
  Identifier id;
  IdExpression(const std::string &name = "") : id(name) {
  }
  IdExpression(const Identifier &id) : id(id) {
  }

  void type_check() override {
  }

  void serialize(std::ostream &ss) override {
    ss << id.name();
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

  void type_check() override;

  void serialize(std::ostream &ss) override;

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

  void type_check() override;

  void serialize(std::ostream &ss) override;

  void flatten(FlattenContext *ctx) override;
};

class ConstExpression : public Expression {
 public:
  TypedConstant val;

  template <typename T>
  ConstExpression(const T &x) : val(x) {
    ret_type = val.dt;
  }

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << val.stringify();
  }

  void flatten(FlattenContext *ctx) override;
};

class ExternalTensorShapeAlongAxisExpression : public Expression {
 public:
  Expr ptr;
  int axis;

  void serialize(std::ostream &ss) override {
    ss << "external_tensor_shape_along_axis(";
    ptr->serialize(ss);
    ss << ", " << axis << ')';
  }

  ExternalTensorShapeAlongAxisExpression(const Expr &ptr, int axis)
      : ptr(ptr), axis(axis) {
  }

  void type_check() override;

  void flatten(FlattenContext *ctx) override;
};

class FuncCallExpression : public Expression {
 public:
  Function *func;
  ExprGroup args;

  void type_check() override;

  void serialize(std::ostream &ss) override;

  FuncCallExpression(Function *func, const ExprGroup &args)
      : func(func), args(args) {
  }

  void flatten(FlattenContext *ctx) override;
};

// Mesh related.

class MeshPatchIndexExpression : public Expression {
 public:
  MeshPatchIndexExpression() {
  }

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << fmt::format("mesh_patch_idx()");
  }

  void flatten(FlattenContext *ctx) override;
};

class MeshRelationAccessExpression : public Expression {
 public:
  mesh::Mesh *mesh;
  Expr mesh_idx;
  mesh::MeshElementType to_type;
  Expr neighbor_idx;

  void type_check() override;

  void serialize(std::ostream &ss) override {
    if (neighbor_idx) {
      ss << "mesh_relation_access(";
      mesh_idx->serialize(ss);
      ss << ", " << mesh::element_type_name(to_type) << "[";
      neighbor_idx->serialize(ss);
      ss << "])";
    } else {
      ss << "mesh_relation_size(";
      mesh_idx->serialize(ss);
      ss << ", " << mesh::element_type_name(to_type) << ")";
    }
  }

  MeshRelationAccessExpression(mesh::Mesh *mesh,
                               const Expr mesh_idx,
                               mesh::MeshElementType to_type)
      : mesh(mesh), mesh_idx(mesh_idx), to_type(to_type) {
  }

  MeshRelationAccessExpression(mesh::Mesh *mesh,
                               const Expr mesh_idx,
                               mesh::MeshElementType to_type,
                               const Expr neighbor_idx)
      : mesh(mesh),
        mesh_idx(mesh_idx),
        to_type(to_type),
        neighbor_idx(neighbor_idx) {
  }

  void flatten(FlattenContext *ctx) override;
};

class MeshIndexConversionExpression : public Expression {
 public:
  mesh::Mesh *mesh;
  mesh::MeshElementType idx_type;
  Expr idx;
  mesh::ConvType conv_type;

  void type_check() override;

  void serialize(std::ostream &ss) override {
    ss << "mesh_index_conversion(" << mesh::conv_type_name(conv_type) << ", "
       << mesh::element_type_name(idx_type) << ", ";
    idx->serialize(ss);
    ss << ")";
  }

  MeshIndexConversionExpression(mesh::Mesh *mesh,
                                mesh::MeshElementType idx_type,
                                const Expr idx,
                                mesh::ConvType conv_type)
      : mesh(mesh), idx_type(idx_type), idx(idx), conv_type(conv_type) {
  }

  void flatten(FlattenContext *ctx) override;
};

class ASTBuilder {
 private:
  std::vector<Block *> stack_;
  Arch arch_;

 public:
  ASTBuilder(Block *initial, Arch arch) : arch_(arch) {
    stack_.push_back(initial);
  }

  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

  struct ScopeGuard {
    ASTBuilder *builder;
    Block *list;
    ScopeGuard(ASTBuilder *builder, Block *list)
        : builder(builder), list(list) {
      builder->stack_.push_back(list);
    }

    ~ScopeGuard() {
      builder->stack_.pop_back();
    }
  };

  // The function will be removed soon
  Arch arch() const {
    return arch_;
  }

  std::unique_ptr<ScopeGuard> create_scope(std::unique_ptr<Block> &list);
  Block *current_block();
  Stmt *get_last_stmt();
  void stop_gradient(SNode *);
  void insert_assignment(Expr &lhs, const Expr &rhs);
  Expr make_var(const Expr &x);
  void insert_for(const Expr &s,
                  const Expr &e,
                  const std::function<void(Expr)> &func);
};

ASTBuilder &current_ast_builder();

class FrontendContext {
 private:
  std::unique_ptr<ASTBuilder> current_builder_;
  std::unique_ptr<Block> root_node_;

 public:
  FrontendContext(Arch arch);

  ASTBuilder &builder() {
    return *current_builder_;
  }

  IRNode *root();

  std::unique_ptr<Block> get_root() {
    return std::move(root_node_);
  }
};

void flatten_lvalue(Expr expr, Expression::FlattenContext *ctx);

void flatten_rvalue(Expr expr, Expression::FlattenContext *ctx);

TLANG_NAMESPACE_END
