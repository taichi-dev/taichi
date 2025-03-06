#pragma once

#include <string>
#include <vector>

#include "taichi/ir/snode_types.h"
#include "taichi/ir/stmt_op_types.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expression.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/function.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/type_system.h"

namespace taichi::lang {

class ASTBuilder;

struct ForLoopConfig {
  bool is_bit_vectorized{false};
  int num_cpu_threads{0};
  bool strictly_serialized{false};
  MemoryAccessOptions mem_access_opt;
  int block_dim{0};
  bool uniform{false};
};

#define TI_DEFINE_CLONE_FOR_FRONTEND_IR                \
  std::unique_ptr<Stmt> clone() const override {       \
    std::unique_ptr<Stmt> new_stmt{                    \
        new std::decay<decltype(*this)>::type{*this}}; \
    new_stmt->ret_type = ret_type;                     \
    return new_stmt;                                   \
  }

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
                           const std::vector<Expr> &outputs,
                           const DebugInfo &dbg_info)
      : Stmt(dbg_info),
        so_func(so_func),
        asm_source(asm_source),
        bc_filename(bc_filename),
        bc_funcname(bc_funcname),
        args(args),
        outputs(outputs) {
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendExprStmt : public Stmt {
 public:
  Expr val;

  explicit FrontendExprStmt(const Expr &val) : val(val) {
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendAllocaStmt : public Stmt {
 public:
  Identifier ident;

  FrontendAllocaStmt(const Identifier &lhs,
                     DataType type,
                     const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info), ident(lhs), is_shared(false) {
    ret_type = type;
  }

  FrontendAllocaStmt(const Identifier &lhs,
                     std::vector<int> shape,
                     DataType element,
                     bool is_shared = false,
                     const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info), ident(lhs), is_shared(is_shared) {
    ret_type = TypeFactory::get_instance().get_pointer_type(
        DataType(TypeFactory::create_tensor_type(shape, element)));
  }

  bool is_shared;

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendSNodeOpStmt : public Stmt {
 public:
  SNodeOpType op_type;
  SNode *snode;
  ExprGroup indices;
  Expr val;

  FrontendSNodeOpStmt(
      SNodeOpType op_type,
      SNode *snode,
      const ExprGroup &indices,
      const Expr &val = Expr(std::shared_ptr<Expression>(nullptr)),
      const DebugInfo &dbg_info = DebugInfo());

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendAssertStmt : public Stmt {
 public:
  std::string text;
  Expr cond;
  std::vector<Expr> args;

  FrontendAssertStmt(const Expr &cond,
                     const std::string &text,
                     const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info), text(text), cond(cond) {
  }

  FrontendAssertStmt(const Expr &cond,
                     const std::string &text,
                     const std::vector<Expr> &args_,
                     const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info), text(text), cond(cond) {
    for (auto &a : args_) {
      args.push_back(a);
    }
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendAssignStmt : public Stmt {
 public:
  Expr lhs, rhs;

  FrontendAssignStmt(const Expr &lhs,
                     const Expr &rhs,
                     const DebugInfo &dbg_info = DebugInfo());

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendIfStmt : public Stmt {
 public:
  Expr condition;
  std::unique_ptr<Block> true_statements, false_statements;

  explicit FrontendIfStmt(const Expr &condition, const DebugInfo &dbg_info)
      : Stmt(dbg_info), condition(condition) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
 private:
  FrontendIfStmt(const FrontendIfStmt &o);
};

class FrontendPrintStmt : public Stmt {
 public:
  using EntryType = std::variant<Expr, std::string>;
  using FormatType = std::optional<std::string>;
  const std::vector<EntryType> contents;
  const std::vector<FormatType> formats;

  FrontendPrintStmt(const std::vector<EntryType> &contents_,
                    const std::vector<FormatType> &formats_,
                    const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info), contents(contents_), formats(formats_) {
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendForStmt : public Stmt {
 public:
  SNode *snode{nullptr};
  Expr external_tensor;
  mesh::Mesh *mesh{nullptr};
  mesh::MeshElementType element_type;
  Expr begin, end;
  std::unique_ptr<Block> body;
  std::vector<Identifier> loop_var_ids;
  bool is_bit_vectorized;
  int num_cpu_threads;
  bool strictly_serialized;
  MemoryAccessOptions mem_access_opt;
  int block_dim;

  FrontendForStmt(const ExprGroup &loop_vars,
                  SNode *snode,
                  Arch arch,
                  const ForLoopConfig &config,
                  const DebugInfo &dbg_info = DebugInfo());

  FrontendForStmt(const ExprGroup &loop_vars,
                  const Expr &external_tensor,
                  Arch arch,
                  const ForLoopConfig &config,
                  const DebugInfo &dbg_info = DebugInfo());

  FrontendForStmt(const ExprGroup &loop_vars,
                  const mesh::MeshPtr &mesh,
                  const mesh::MeshElementType &element_type,
                  Arch arch,
                  const ForLoopConfig &config,
                  const DebugInfo &dbg_info = DebugInfo());

  FrontendForStmt(const Expr &loop_var,
                  const Expr &begin,
                  const Expr &end,
                  Arch arch,
                  const ForLoopConfig &config,
                  const DebugInfo &dbg_info = DebugInfo());

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR

 private:
  FrontendForStmt(const FrontendForStmt &o);

  void init_config(Arch arch, const ForLoopConfig &config);

  void init_loop_vars(const ExprGroup &loop_vars);

  void add_loop_var(const Expr &loop_var);
};

class FrontendFuncDefStmt : public Stmt {
 public:
  std::string funcid;
  std::unique_ptr<Block> body;

  explicit FrontendFuncDefStmt(const std::string &funcid) : funcid(funcid) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR

 private:
  FrontendFuncDefStmt(const FrontendFuncDefStmt &o);
};

class FrontendBreakStmt : public Stmt {
 public:
  explicit FrontendBreakStmt(const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info) {
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendContinueStmt : public Stmt {
 public:
  explicit FrontendContinueStmt(const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info) {
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class FrontendWhileStmt : public Stmt {
 public:
  Expr cond;
  std::unique_ptr<Block> body;

  explicit FrontendWhileStmt(const Expr &cond, const DebugInfo &dbg_info)
      : Stmt(dbg_info), cond(cond) {
  }

  bool is_container_statement() const override {
    return true;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
 private:
  FrontendWhileStmt(const FrontendWhileStmt &o);
};

class FrontendReturnStmt : public Stmt {
 public:
  ExprGroup values;

  explicit FrontendReturnStmt(const ExprGroup &group,
                              const DebugInfo &dbg_info = DebugInfo());

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

// Expressions

class ArgLoadExpression : public Expression {
 public:
  const std::vector<int> arg_id;
  DataType dt;
  bool is_ptr;

  /* Creates a load statement if true, otherwise returns the pointer
   * directly.
   * TODO: Split ArgLoad into two steps: ArgAddr and GlobalLoad.
   */
  bool create_load;

  int arg_depth;

  ArgLoadExpression(const std::vector<int> &arg_id,
                    DataType dt,
                    bool is_ptr = false,
                    bool create_load = true,
                    int arg_depth = 0,
                    const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info),
        arg_id(arg_id),
        dt(dt),
        is_ptr(is_ptr),
        create_load(create_load),
        arg_depth(arg_depth) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  bool is_lvalue() const override {
    return is_ptr;
  }

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class Texture;

class TexturePtrExpression : public Expression {
 public:
  const std::vector<int> arg_id;
  int num_dims;
  bool is_storage{false};
  int arg_depth;

  // Optional, for storage textures
  BufferFormat format{BufferFormat::unknown};
  int lod{0};

  explicit TexturePtrExpression(const std::vector<int> &arg_id,
                                int num_dims,
                                int arg_depth,
                                const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info),
        arg_id(arg_id),
        num_dims(num_dims),
        is_storage(false),
        arg_depth(arg_depth),
        format(BufferFormat::rgba8),
        lod(0) {
  }

  TexturePtrExpression(const std::vector<int> &arg_id,
                       int num_dims,
                       int arg_depth,
                       BufferFormat format,
                       int lod,
                       const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info),
        arg_id(arg_id),
        num_dims(num_dims),
        is_storage(true),
        arg_depth(arg_depth),
        format(format),
        lod(lod) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class RandExpression : public Expression {
 public:
  DataType dt;

  explicit RandExpression(DataType dt, const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), dt(dt) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class UnaryOpExpression : public Expression {
 public:
  UnaryOpType type;
  Expr operand;
  DataType cast_type;

  UnaryOpExpression(UnaryOpType type,
                    const Expr &operand,
                    const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), type(type), operand(operand) {
    cast_type = PrimitiveType::unknown;
  }

  UnaryOpExpression(UnaryOpType type,
                    const Expr &operand,
                    DataType cast_type,
                    const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info),
        type(type),
        operand(operand),
        cast_type(cast_type) {
  }

  void type_check(const CompileConfig *config) override;

  bool is_cast() const;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class BinaryOpExpression : public Expression {
 public:
  BinaryOpType type;
  Expr lhs, rhs;

  BinaryOpExpression(const BinaryOpType &type, const Expr &lhs, const Expr &rhs)
      : type(type), lhs(lhs), rhs(rhs) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
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

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class InternalFuncCallExpression : public Expression {
 public:
  Operation *op;
  std::vector<Expr> args;

  InternalFuncCallExpression(Operation *op, const std::vector<Expr> &args_)
      : op(op), args(args_) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

// TODO: Make this a non-expr
class ExternalTensorExpression : public Expression {
 public:
  DataType dt;
  int ndim;
  std::vector<int> arg_id;
  bool needs_grad{false};
  bool is_grad{false};
  int arg_depth{0};
  BoundaryMode boundary{BoundaryMode::kUnsafe};

  ExternalTensorExpression(const DataType &dt,
                           int ndim,
                           const std::vector<int> &arg_id,
                           bool needs_grad = false,
                           int arg_depth = false,
                           BoundaryMode boundary = BoundaryMode::kUnsafe) {
    init(dt, ndim, arg_id, needs_grad, arg_depth, boundary);
  }

  explicit ExternalTensorExpression(Expr *expr) : is_grad(true) {
    auto ptr = expr->cast<ExternalTensorExpression>();
    init(ptr->dt, ptr->ndim, ptr->arg_id, ptr->needs_grad, ptr->arg_depth,
         ptr->boundary);
  }

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION

  const CompileConfig *get_compile_config() {
    TI_ASSERT(config_ != nullptr);
    return config_;
  }

  void type_check(const CompileConfig *config) override {
    ret_type = TypeFactory::get_instance().get_ndarray_struct_type(dt, ndim,
                                                                   needs_grad);
    ret_type.set_is_pointer(true);
    config_ = config;
  }

 private:
  const CompileConfig *config_ = nullptr;

  void init(const DataType &dt,
            int ndim,
            const std::vector<int> &arg_id,
            bool needs_grad,
            int arg_depth,
            BoundaryMode boundary) {
    this->dt = dt;
    this->ndim = ndim;
    this->arg_id = arg_id;
    this->needs_grad = needs_grad;
    this->arg_depth = arg_depth;
    this->boundary = boundary;
  }
};

// TODO: Make this a non-expr
class FieldExpression : public Expression {
 public:
  Identifier ident;
  DataType dt;
  std::string name;
  SNode *snode{nullptr};
  SNodeGradType snode_grad_type{SNodeGradType::kPrimal};
  bool has_ambient{false};
  TypedConstant ambient_value;
  Expr adjoint;
  Expr dual;
  Expr adjoint_checkbit;

  FieldExpression(DataType dt, const Identifier &ident) : ident(ident), dt(dt) {
  }

  void type_check(const CompileConfig *config) override {
  }

  void set_snode(SNode *snode) {
    this->snode = snode;
  }

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class MatrixFieldExpression : public Expression {
 public:
  std::vector<Expr> fields;
  std::vector<int> element_shape;
  bool dynamic_indexable{false};
  int dynamic_index_stride{0};

  MatrixFieldExpression(const std::vector<Expr> &fields,
                        const std::vector<int> &element_shape)
      : fields(fields), element_shape(element_shape) {
    for (auto &field : fields) {
      TI_ASSERT(field.is<FieldExpression>());
    }
    TI_ASSERT(!fields.empty());
    auto compute_type =
        fields[0].cast<FieldExpression>()->dt->get_compute_type();
    for (auto &field : fields) {
      if (field.cast<FieldExpression>()->dt->get_compute_type() !=
          compute_type) {
        throw TaichiRuntimeError(
            "Member fields of a matrix field must have the same compute type");
      }
    }
  }

  void type_check(const CompileConfig *config) override {
  }

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

/**
 * Creating a local matrix;
 * lowered from ti.Matrix
 */
class MatrixExpression : public Expression {
 public:
  std::vector<Expr> elements;
  DataType dt;

  MatrixExpression(const std::vector<Expr> &elements,
                   std::vector<int> shape,
                   DataType element_type,
                   const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), elements(elements) {
    dt = TypeFactory::create_tensor_type(shape, element_type);
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class IndexExpression : public Expression {
 public:
  // `var` is one of FieldExpression, MatrixFieldExpression,
  // ExternalTensorExpression, IdExpression
  Expr var;
  // In the cases of matrix slice and vector swizzle, there can be multiple
  // indices, and the corresponding ret_shape should also be recorded. In normal
  // index expressions ret_shape will be left empty.
  std::vector<ExprGroup> indices_group;
  std::vector<int> ret_shape;

  IndexExpression(const Expr &var,
                  const ExprGroup &indices,
                  const DebugInfo &dbg_info = DebugInfo());

  IndexExpression(const Expr &var,
                  const std::vector<ExprGroup> &indices_group,
                  const std::vector<int> &ret_shape,
                  const DebugInfo &dbg_info = DebugInfo());

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  bool is_lvalue() const override {
    return true;
  }

  // whether the LocalLoad/Store or GlobalLoad/Store is to be used on the
  // compiled stmt
  bool is_local() const;
  bool is_global() const;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION

 private:
  bool is_field() const;
  bool is_matrix_field() const;
  bool is_ndarray() const;
  bool is_tensor() const;
};

class RangeAssumptionExpression : public Expression {
 public:
  Expr input, base;
  int low, high;

  RangeAssumptionExpression(const Expr &input,
                            const Expr &base,
                            int low,
                            int high,
                            const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), input(input), base(base), low(low), high(high) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class LoopUniqueExpression : public Expression {
 public:
  Expr input;
  std::vector<SNode *> covers;

  LoopUniqueExpression(const Expr &input,
                       const std::vector<SNode *> &covers,
                       const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), input(input), covers(covers) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class IdExpression : public Expression {
 public:
  Identifier id;

  explicit IdExpression(const Identifier &id) : id(id) {
  }

  void type_check(const CompileConfig *config) override {
  }

  void flatten(FlattenContext *ctx) override;

  Stmt *flatten_noload(FlattenContext *ctx) {
    return ctx->current_block->lookup_var(id);
  }

  bool is_lvalue() const override {
    return true;
  }

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

// ti.atomic_*() is an expression with side effect.
class AtomicOpExpression : public Expression {
 public:
  AtomicOpType op_type;
  Expr dest, val;

  AtomicOpExpression(AtomicOpType op_type, const Expr &dest, const Expr &val)
      : op_type(op_type), dest(dest), val(val) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class SNodeOpExpression : public Expression {
 public:
  SNode *snode;
  SNodeOpType op_type;
  ExprGroup indices;
  std::vector<Expr> values;  // Only for op_type==append

  SNodeOpExpression(SNode *snode,
                    SNodeOpType op_type,
                    const ExprGroup &indices);

  SNodeOpExpression(SNode *snode,
                    SNodeOpType op_type,
                    const ExprGroup &indices,
                    const std::vector<Expr> &values);

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class TextureOpExpression : public Expression {
 public:
  TextureOpType op;
  Expr texture_ptr;
  ExprGroup args;

  explicit TextureOpExpression(TextureOpType op,
                               Expr texture_ptr,
                               const ExprGroup &args,
                               const DebugInfo &dbg_info = DebugInfo());

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class ConstExpression : public Expression {
 public:
  TypedConstant val;

  template <typename T>
  explicit ConstExpression(const T &x) : val(x) {
    ret_type = val.dt;
  }
  template <typename T>
  ConstExpression(const DataType &dt, const T &x) : val({dt, x}) {
    ret_type = dt;
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class ExternalTensorShapeAlongAxisExpression : public Expression {
 public:
  Expr ptr;
  int axis;

  ExternalTensorShapeAlongAxisExpression(
      const Expr &ptr,
      int axis,
      const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), ptr(ptr), axis(axis) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class ExternalTensorBasePtrExpression : public Expression {
 public:
  Expr ptr;
  bool is_grad;

  explicit ExternalTensorBasePtrExpression(
      const Expr &ptr,
      bool is_grad,
      const DebugInfo &dbg_info = DebugInfo())
      : ptr(ptr), is_grad(is_grad) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class FrontendFuncCallStmt : public Stmt {
 public:
  std::optional<Identifier> ident;
  Function *func;
  ExprGroup args;

  explicit FrontendFuncCallStmt(
      Function *func,
      const ExprGroup &args,
      const std::optional<Identifier> &id = std::nullopt,
      const DebugInfo &dbg_info = DebugInfo())
      : Stmt(dbg_info), ident(id), func(func), args(args) {
    TI_ASSERT(id.has_value() == !func->rets.empty());
  }

  bool is_container_statement() const override {
    return false;
  }

  TI_DEFINE_ACCEPT
  TI_DEFINE_CLONE_FOR_FRONTEND_IR
};

class GetElementExpression : public Expression {
 public:
  Expr src;
  std::vector<int> index;

  void type_check(const CompileConfig *config) override;

  GetElementExpression(const Expr &src,
                       std::vector<int> index,
                       const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), src(src), index(index) {
  }

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

// Mesh related.

class MeshPatchIndexExpression : public Expression {
 public:
  explicit MeshPatchIndexExpression(const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info) {
  }

  void type_check(const CompileConfig *config) override;

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class MeshRelationAccessExpression : public Expression {
 public:
  mesh::Mesh *mesh;
  Expr mesh_idx;
  mesh::MeshElementType to_type;
  Expr neighbor_idx;

  void type_check(const CompileConfig *config) override;

  MeshRelationAccessExpression(mesh::Mesh *mesh,
                               const Expr mesh_idx,
                               mesh::MeshElementType to_type,
                               const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), mesh(mesh), mesh_idx(mesh_idx), to_type(to_type) {
  }

  MeshRelationAccessExpression(mesh::Mesh *mesh,
                               const Expr mesh_idx,
                               mesh::MeshElementType to_type,
                               const Expr neighbor_idx,
                               const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info),
        mesh(mesh),
        mesh_idx(mesh_idx),
        to_type(to_type),
        neighbor_idx(neighbor_idx) {
  }

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class MeshIndexConversionExpression : public Expression {
 public:
  mesh::Mesh *mesh;
  mesh::MeshElementType idx_type;
  Expr idx;
  mesh::ConvType conv_type;

  void type_check(const CompileConfig *config) override;

  MeshIndexConversionExpression(mesh::Mesh *mesh,
                                mesh::MeshElementType idx_type,
                                const Expr idx,
                                mesh::ConvType conv_type,
                                const DebugInfo &dbg_info = DebugInfo());

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class ReferenceExpression : public Expression {
 public:
  Expr var;
  void type_check(const CompileConfig *config) override;

  explicit ReferenceExpression(const Expr &expr,
                               const DebugInfo &dbg_info = DebugInfo())
      : Expression(dbg_info), var(expr) {
  }

  void flatten(FlattenContext *ctx) override;

  TI_DEFINE_ACCEPT_FOR_EXPRESSION
};

class ASTBuilder {
 private:
  enum LoopState { None, Outermost, Inner };
  enum LoopType { NotLoop, For, While };

  class ForLoopDecoratorRecorder {
   public:
    ForLoopConfig config;

    ForLoopDecoratorRecorder() {
      reset();
    }

    void reset() {
      config.is_bit_vectorized = false;
      config.num_cpu_threads = 0;
      config.uniform = false;
      config.mem_access_opt.clear();
      config.block_dim = 0;
      config.strictly_serialized = false;
    }
  };

  std::vector<Block *> stack_;
  std::vector<LoopState> loop_state_stack_;
  bool is_kernel_{false};
  Arch arch_;
  ForLoopDecoratorRecorder for_loop_dec_;
  int id_counter_{0};

 public:
  ASTBuilder(Block *initial, Arch arch, bool is_kernel)
      : is_kernel_(is_kernel), arch_(arch) {
    stack_.push_back(initial);
    loop_state_stack_.push_back(None);
  }

  void insert(std::unique_ptr<Stmt> &&stmt, int location = -1);

  Block *current_block();
  Stmt *get_last_stmt();
  void stop_gradient(SNode *);
  void insert_assignment(Expr &lhs,
                         const Expr &rhs,
                         const DebugInfo &dbg_info = DebugInfo());
  Expr make_var(const Expr &x, const DebugInfo &dbg_info = DebugInfo());
  void insert_for(const Expr &s,
                  const Expr &e,
                  const std::function<void(Expr)> &func);

  Expr make_id_expr(const std::string &name);
  Expr make_matrix_expr(const std::vector<int> &shape,
                        const DataType &dt,
                        const std::vector<Expr> &elements,
                        const DebugInfo &dbg_info = DebugInfo());
  Expr insert_thread_idx_expr();
  Expr insert_patch_idx_expr(const DebugInfo &dbg_info = DebugInfo());
  void create_kernel_exprgroup_return(const ExprGroup &group,
                                      const DebugInfo &dbg_info = DebugInfo());
  void create_print(std::vector<std::variant<Expr, std::string>> contents,
                    std::vector<std::optional<std::string>> formats,
                    const DebugInfo &dbg_info = DebugInfo());
  void begin_func(const std::string &funcid);
  void end_func(const std::string &funcid);
  void begin_frontend_if(const Expr &cond,
                         const DebugInfo &dbg_info = DebugInfo());
  void begin_frontend_if_true();
  void begin_frontend_if_false();
  void insert_external_func_call(std::size_t func_addr,
                                 std::string source,
                                 std::string filename,
                                 std::string funcname,
                                 const ExprGroup &args,
                                 const ExprGroup &outputs,
                                 const DebugInfo &dbg_info = DebugInfo());
  Expr expr_alloca(const DebugInfo &dbg_info = DebugInfo());
  Expr expr_alloca_shared_array(const std::vector<int> &shape,
                                const DataType &element_type,
                                const DebugInfo &dbg_info = DebugInfo());
  Expr expr_subscript(const Expr &expr,
                      const ExprGroup &indices,
                      const DebugInfo &dbg_info = DebugInfo());

  Expr mesh_index_conversion(mesh::MeshPtr mesh_ptr,
                             mesh::MeshElementType idx_type,
                             const Expr &idx,
                             mesh::ConvType &conv_type,
                             const DebugInfo &dbg_info = DebugInfo());

  void expr_assign(const Expr &lhs,
                   const Expr &rhs,
                   const DebugInfo &dbg_info = DebugInfo());
  std::optional<Expr> insert_func_call(Function *func,
                                       const ExprGroup &args,
                                       const DebugInfo &dbg_info = DebugInfo());
  void create_assert_stmt(const Expr &cond,
                          const std::string &msg,
                          const std::vector<Expr> &args,
                          const DebugInfo &dbg_info = DebugInfo());
  void begin_frontend_range_for(const Expr &i,
                                const Expr &s,
                                const Expr &e,
                                const DebugInfo &dbg_info = DebugInfo());
  void begin_frontend_struct_for_on_snode(
      const ExprGroup &loop_vars,
      SNode *snode,
      const DebugInfo &dbg_info = DebugInfo());
  void begin_frontend_struct_for_on_external_tensor(
      const ExprGroup &loop_vars,
      const Expr &external_tensor,
      const DebugInfo &dbg_info = DebugInfo());
  void begin_frontend_mesh_for(const Expr &i,
                               const mesh::MeshPtr &mesh_ptr,
                               const mesh::MeshElementType &element_type,
                               const DebugInfo &dbg_info = DebugInfo());
  void begin_frontend_while(const Expr &cond,
                            const DebugInfo &dbg_info = DebugInfo());
  void insert_break_stmt(const DebugInfo &dbg_info = DebugInfo());
  void insert_continue_stmt(const DebugInfo &dbg_info = DebugInfo());
  void insert_expr_stmt(const Expr &val);
  void insert_snode_activate(SNode *snode,
                             const ExprGroup &expr_group,
                             const DebugInfo &dbg_info = DebugInfo());
  void insert_snode_deactivate(SNode *snode,
                               const ExprGroup &expr_group,
                               const DebugInfo &dbg_info = DebugInfo());
  Expr make_texture_op_expr(const TextureOpType &op,
                            const Expr &texture_ptr,
                            const ExprGroup &args,
                            const DebugInfo &dbg_info = DebugInfo());
  /*
   * This function allocates the space for a new item (a struct or a scalar)
   * in the Dynamic SNode, and assigns values to the elements inside it.
   *
   * When appending a struct, the size of vals must be equal to
   * the number of elements in the struct. When appending a scalar,
   * the size of vals must be one.
   */
  Expr snode_append(SNode *snode,
                    const ExprGroup &indices,
                    const std::vector<Expr> &vals);
  Expr snode_is_active(SNode *snode, const ExprGroup &indices);
  Expr snode_length(SNode *snode, const ExprGroup &indices);
  Expr snode_get_addr(SNode *snode, const ExprGroup &indices);

  std::vector<Expr> expand_exprs(const std::vector<Expr> &exprs);

  void create_scope(std::unique_ptr<Block> &list, LoopType tp = NotLoop);
  void pop_scope();

  void bit_vectorize() {
    for_loop_dec_.config.is_bit_vectorized = true;
  }

  void parallelize(int v) {
    for_loop_dec_.config.num_cpu_threads = v;
  }

  void strictly_serialize() {
    for_loop_dec_.config.strictly_serialized = true;
  }

  void block_dim(int v) {
    if (arch_ == Arch::cuda || arch_ == Arch::vulkan || arch_ == Arch::amdgpu) {
      TI_ASSERT((v % 32 == 0) || bit::is_power_of_two(v));
    } else {
      TI_ASSERT(bit::is_power_of_two(v));
    }
    for_loop_dec_.config.block_dim = v;
  }

  void insert_snode_access_flag(SNodeAccessFlag v, const Expr &field) {
    for_loop_dec_.config.mem_access_opt.add_flag(field.snode(), v);
  }

  void reset_snode_access_flag() {
    for_loop_dec_.reset();
  }

  Identifier get_next_id(const std::string &name = "") {
    return Identifier(id_counter_++, name);
  }
};

class FrontendContext {
 private:
  std::unique_ptr<ASTBuilder> current_builder_;
  std::unique_ptr<Block> root_node_;

 public:
  explicit FrontendContext(Arch arch, bool is_kernel) {
    root_node_ = std::make_unique<Block>();
    current_builder_ =
        std::make_unique<ASTBuilder>(root_node_.get(), arch, is_kernel);
  }

  ASTBuilder &builder() {
    return *current_builder_;
  }

  std::unique_ptr<Block> get_root() {
    return std::move(root_node_);
  }
};

Stmt *flatten_lvalue(Expr expr, Expression::FlattenContext *ctx);

Stmt *flatten_rvalue(Expr expr, Expression::FlattenContext *ctx);

}  // namespace taichi::lang
