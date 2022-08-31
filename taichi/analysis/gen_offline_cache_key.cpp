#include "offline_cache_util.h"

#include "taichi/ir/expr.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/type.h"
#include "taichi/program/function.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

namespace {

enum class ExprOpCode : std::uint8_t {
  NIL,
#define PER_EXPRESSION(x) x,
#include "taichi/inc/expressions.inc.h"
#undef PER_EXPRESSION
};

enum class StmtOpCode : std::uint8_t {
  NIL,
  EnterBlock,
  ExitBlock,
  StopGrad,
#define PER_STATEMENT(x) x,
#include "taichi/inc/frontend_statements.inc.h"
#undef PER_STATEMENT
};

enum class ForLoopType : std::uint8_t {
  RangeFor,
  StructFor,
  MeshFor,
};

enum class ExternalFuncType : std::uint8_t {
  SO,
  ASM,
  BC,
};

enum class MeshRelationAccessType {
  Access,  // mesh_relation_access
  Size,    // mesh_relation_size
};

class ASTSerializer : public IRVisitor, public ExpressionVisitor {
 private:
  using ExpressionVisitor::visit;
  using IRVisitor::visit;

 public:
  ASTSerializer(Program *prog, std::ostream *os)
      : ExpressionVisitor(true), prog_(prog), os_(os) {
    this->allow_undefined_visitor = true;
  }

  void set_ostream(std::ostream *os) {
    this->os_ = os;
  }

  std::ostream *get_ostream() {
    return this->os_;
  }

  void visit(Expression *expr) override {
    this->ExpressionVisitor::visit(expr);
  }

  void visit(Stmt *stmt) override {
    this->IRVisitor::visit(stmt);
  }

  void visit(ExprGroup &expr_group) override {
    emit(expr_group.exprs);
  }

  void visit(ArgLoadExpression *expr) override {
    emit(ExprOpCode::ArgLoadExpression);
    emit(expr->dt);
    emit(expr->arg_id);
    emit(expr->is_ptr);
  }

  void visit(TexturePtrExpression *expr) override {
    emit(ExprOpCode::TexturePtrExpression);
    emit(expr->arg_id);
    emit(expr->num_dims);
    emit(expr->is_storage);
    emit(expr->num_channels);
    emit(expr->channel_format);
    emit(expr->lod);
  }

  void visit(TextureOpExpression *expr) override {
    emit(ExprOpCode::TextureOpExpression);
    emit(expr->op);
    emit(expr->texture_ptr);
    emit(expr->args.exprs);
  }

  void visit(RandExpression *expr) override {
    emit(ExprOpCode::RandExpression);
    emit(expr->dt);
  }

  void visit(UnaryOpExpression *expr) override {
    emit(ExprOpCode::UnaryOpExpression);
    emit(expr->type);
    if (expr->is_cast()) {
      emit(expr->cast_type);
    }
    emit(expr->operand);
  }

  void visit(BinaryOpExpression *expr) override {
    emit(ExprOpCode::BinaryOpExpression);
    emit(expr->type);
    emit(expr->lhs);
    emit(expr->rhs);
  }

  void visit(TernaryOpExpression *expr) override {
    emit(ExprOpCode::TernaryOpExpression);
    emit(expr->type);
    emit(expr->op1);
    emit(expr->op2);
    emit(expr->op3);
  }

  void visit(InternalFuncCallExpression *expr) override {
    emit(ExprOpCode::InternalFuncCallExpression);
    emit(expr->func_name);
    emit(expr->args);
    emit(expr->with_runtime_context);
  }

  void visit(ExternalTensorExpression *expr) override {
    emit(ExprOpCode::ExternalTensorExpression);
    emit(expr->dt);
    emit(expr->dim);
    emit(expr->arg_id);
    emit(expr->element_dim);
  }

  void visit(GlobalVariableExpression *expr) override {
    emit(ExprOpCode::GlobalVariableExpression);
    emit(expr->ident);
    emit(expr->dt);
    emit(expr->snode);
    emit(expr->has_ambient);
    emit(expr->ambient_value);
    emit(expr->snode_grad_type);
    emit(expr->adjoint);
    emit(expr->dual);
    emit(expr->adjoint_checkbit);
  }

  void visit(IndexExpression *expr) override {
    emit(ExprOpCode::IndexExpression);
    emit(expr->var);
    emit(expr->indices.exprs);
  }

  void visit(MatrixExpression *expr) override {
    emit(ExprOpCode::MatrixExpression);
    emit(expr->dt);
    for (auto elt : expr->elements) {
      emit(elt);
    }
  }

  void visit(StrideExpression *expr) override {
    emit(ExprOpCode::StrideExpression);
    emit(expr->var);
    emit(expr->indices.exprs);
    emit(expr->shape);
    emit(expr->stride);
  }

  void visit(RangeAssumptionExpression *expr) override {
    emit(ExprOpCode::RangeAssumptionExpression);
    emit(expr->input);
    emit(expr->base);
    emit(expr->low);
    emit(expr->high);
  }

  void visit(LoopUniqueExpression *expr) override {
    emit(ExprOpCode::LoopUniqueExpression);
    emit(expr->input);
    emit(expr->covers);
  }

  void visit(IdExpression *expr) override {
    emit(ExprOpCode::IdExpression);
    emit(expr->id);
  }

  void visit(AtomicOpExpression *expr) override {
    emit(ExprOpCode::AtomicOpExpression);
    emit(expr->op_type);
    emit(expr->dest);
    emit(expr->val);
  }

  void visit(SNodeOpExpression *expr) override {
    emit(ExprOpCode::SNodeOpExpression);
    emit(expr->op_type);
    emit(expr->snode);
    emit(expr->indices.exprs);
    emit(expr->value);
  }

  void visit(ConstExpression *expr) override {
    emit(ExprOpCode::ConstExpression);
    emit(expr->val);
  }

  void visit(ExternalTensorShapeAlongAxisExpression *expr) override {
    emit(ExprOpCode::ExternalTensorShapeAlongAxisExpression);
    emit(expr->ptr);
    emit(expr->axis);
  }

  void visit(FuncCallExpression *expr) override {
    emit(ExprOpCode::FuncCallExpression);
    emit(expr->func);
    emit(expr->args.exprs);
  }

  void visit(MeshPatchIndexExpression *expr) override {
    emit(ExprOpCode::MeshPatchIndexExpression);
  }

  void visit(MeshRelationAccessExpression *expr) override {
    emit(ExprOpCode::MeshRelationAccessExpression);
    if (expr->neighbor_idx) {
      emit(MeshRelationAccessType::Access);
      emit(expr->neighbor_idx);
    } else {
      emit(MeshRelationAccessType::Size);
    }
    emit(expr->mesh);
    emit(expr->to_type);
    emit(expr->mesh_idx);
  }

  void visit(MeshIndexConversionExpression *expr) override {
    emit(ExprOpCode::MeshIndexConversionExpression);
    emit(expr->mesh);
    emit(expr->idx_type);
    emit(expr->idx);
    emit(expr->conv_type);
  }

  void visit(ReferenceExpression *expr) override {
    emit(ExprOpCode::ReferenceExpression);
    emit(expr->var);
  }

  void visit(Block *block) override {
    emit(StmtOpCode::EnterBlock);
    emit(static_cast<std::size_t>(block->statements.size()));
    for (auto &stmt : block->statements) {
      emit(stmt.get());
    }
    emit(StmtOpCode::StopGrad);
    emit(block->stop_gradients);
    emit(StmtOpCode::ExitBlock);
  }

  void visit(FrontendExprStmt *stmt) override {
    emit(StmtOpCode::FrontendExprStmt);
    emit(stmt->val);
  }

  void visit(FrontendBreakStmt *stmt) override {
    emit(StmtOpCode::FrontendBreakStmt);
  }

  void visit(FrontendContinueStmt *stmt) override {
    emit(StmtOpCode::FrontendContinueStmt);
  }

  void visit(FrontendAssignStmt *stmt) override {
    emit(StmtOpCode::FrontendAssignStmt);
    emit(stmt->lhs);
    emit(stmt->rhs);
  }

  void visit(FrontendAllocaStmt *stmt) override {
    emit(StmtOpCode::FrontendAllocaStmt);
    emit(stmt->ident);
  }

  void visit(FrontendAssertStmt *stmt) override {
    emit(StmtOpCode::FrontendAssertStmt);
    emit(stmt->cond);
    emit(stmt->text);
    emit(stmt->args);
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    emit(StmtOpCode::FrontendSNodeOpStmt);
    emit(stmt->op_type);
    emit(stmt->snode);
    emit(stmt->indices.exprs);
    emit(stmt->val);
  }

  void visit(FrontendIfStmt *stmt) override {
    emit(StmtOpCode::FrontendIfStmt);
    emit(stmt->condition);
    std::uint8_t branch_count = 0;
    if (stmt->true_statements) {
      ++branch_count;
    }
    if (stmt->false_statements) {
      ++branch_count;
    }
    emit(branch_count);
    if (stmt->true_statements) {
      emit(stmt->true_statements.get());
    }
    if (stmt->false_statements) {
      emit(stmt->false_statements.get());
    }
  }

  void visit(FrontendPrintStmt *stmt) override {
    emit(StmtOpCode::FrontendPrintStmt);
    emit(static_cast<std::size_t>(stmt->contents.size()));
    for (const auto &c : stmt->contents) {
      emit(static_cast<std::uint8_t>(c.index()));
      if (std::holds_alternative<Expr>(c)) {
        emit(std::get<Expr>(c).expr);
      } else {
        emit(std::get<std::string>(c));
      }
    }
  }

  void visit(FrontendFuncDefStmt *stmt) override {
    emit(StmtOpCode::FrontendFuncDefStmt);
    emit(stmt->body.get());
  }

  void visit(FrontendWhileStmt *stmt) override {
    emit(StmtOpCode::FrontendWhileStmt);
    emit(stmt->cond);
    emit(stmt->body.get());
  }

  void visit(FrontendForStmt *stmt) override {
    emit(StmtOpCode::FrontendForStmt);
    if (stmt->is_ranged()) {
      emit(ForLoopType::RangeFor);
      emit(stmt->loop_var_id);
      emit(stmt->begin);
      emit(stmt->end);
    } else if (stmt->mesh_for) {
      emit(ForLoopType::MeshFor);
      emit(stmt->element_type);
      emit(stmt->mesh);
    } else {
      emit(ForLoopType::StructFor);
      emit(stmt->loop_var_id);
      emit(stmt->global_var);
    }
    emit(stmt->is_bit_vectorized);
    emit(stmt->num_cpu_threads);
    emit(stmt->strictly_serialized);
    emit(stmt->mem_access_opt);
    emit(stmt->block_dim);
    emit(stmt->body.get());
  }

  void visit(FrontendReturnStmt *stmt) override {
    emit(StmtOpCode::FrontendReturnStmt);
    emit(stmt->values.exprs);
  }

  void visit(FrontendExternalFuncStmt *stmt) override {
    // Note: The result of serializing FrontendExternalFuncStmt is not parsable
    // now
    emit(StmtOpCode::FrontendExternalFuncStmt);
    if (stmt->so_func != nullptr) {
      emit(ExternalFuncType::SO);
    } else if (!stmt->asm_source.empty()) {
      emit(ExternalFuncType::ASM);
      emit(stmt->asm_source);
    } else {
      emit(ExternalFuncType::BC);
      emit(stmt->bc_filename);
      emit(stmt->bc_funcname);
    }
    emit(stmt->args);
    emit(stmt->outputs);
  }

  static void run(Program *prog, IRNode *ast, std::ostream *os) {
    ASTSerializer serializer(prog, os);
    ast->accept(&serializer);
    serializer.emit_dependencies();
  }

 private:
  void emit_dependencies() {
    // Serialize dependent real-functions
    emit(real_funcs_.size());
    for (auto &[func, id] : real_funcs_) {
      if (auto &ast_str = func->try_get_ast_serialization_data();
          ast_str.has_value()) {
        emit_bytes(ast_str->c_str(), ast_str->size());
      }
    }

    // Serialize snode_trees(Temporary: using offline-cache-key of SNode)
    // Note: The result of serializing snode_tree_roots_ is not parsable now
    emit(static_cast<std::size_t>(snode_tree_roots_.size()));
    for (auto *snode : snode_tree_roots_) {
      auto key = get_hashed_offline_cache_key_of_snode(snode);
      emit_bytes(key.c_str(), key.size());
    }

    // Dump string-pool
    emit(static_cast<std::size_t>(string_pool_.size()));
    emit_bytes(string_pool_.data(), string_pool_.size());
  }

  template <typename T>
  void emit_pod(const T &val) {
    static_assert(std::is_pod<T>::value);
    TI_ASSERT(os_);
    os_->write((const char *)&val, sizeof(T));
  }

  void emit_bytes(const char *bytes, std::size_t len) {
    TI_ASSERT(os_);
    if (!bytes)
      return;
    os_->write(bytes, len);
  }

  template <typename T>
  void emit(const std::vector<T> &v) {
    emit(static_cast<std::size_t>(v.size()));
    for (const auto &e : v) {
      emit(e);
    }
  }

  template <typename K, typename V>
  void emit(const std::unordered_map<K, V> &map) {
    emit(static_cast<std::size_t>(map.size()));
    for (const auto &[k, v] : map) {
      emit(k);
      emit(v);
    }
  }

  template <typename T1, typename T2>
  void emit(const std::pair<T1, T2> &pair) {
    emit(pair.first);
    emit(pair.second);
  }

  template <typename K, typename V>
  void emit(const std::map<K, V> &map) {
    emit(static_cast<std::size_t>(map.size()));
    for (const auto &[k, v] : map) {
      emit(k);
      emit(v);
    }
  }

  void emit(const std::string &str) {
    std::size_t size = str.size();
    std::size_t offset = string_pool_.size();
    string_pool_.insert(string_pool_.end(), str.begin(), str.end());
    emit(size);
    emit(offset);
  }

  void emit(Function *func) {
    TI_ASSERT(func);
    auto iter = real_funcs_.find(func);
    if (iter != real_funcs_.end()) {
      emit(iter->second);
    } else {
      auto [iter, ok] = real_funcs_.insert({func, real_funcs_.size()});
      TI_ASSERT(ok);
      emit(iter->second);
    }
  }

  void emit(const TypedConstant &val) {
    emit(val.dt);
    if (!val.dt->is_primitive(PrimitiveTypeID::unknown)) {
      emit(val.stringify());
    }
  }

  void emit(SNode *snode) {
    TI_ASSERT(prog_);
    if (snode) {
      emit(static_cast<std::size_t>(snode->get_snode_tree_id()));
      emit(static_cast<std::size_t>(snode->id));
      auto *root = prog_->get_snode_root(snode->get_snode_tree_id());
      snode_tree_roots_.insert(root);
    } else {
      emit(std::numeric_limits<std::size_t>::max());
      emit(std::numeric_limits<std::size_t>::max());
    }
  }

  void emit(const mesh::MeshLocalRelation &r) {
    emit(r.fixed);
    emit(r.value);
    emit(r.patch_offset);
    emit(r.offset);
  }

  void emit(mesh::Mesh *mesh) {
    TI_ASSERT(mesh);
    emit(mesh->num_patches);
    emit(mesh->num_elements);
    emit(mesh->patch_max_element_num);
    emit(mesh->owned_offset);
    emit(mesh->total_offset);
    emit(mesh->index_mapping);
    emit(mesh->relations);
  }

  void emit(const Identifier &ident) {
    emit(ident.id);
  }

  void emit(const DataType &type) {
    if (auto *p = type->cast<PrimitiveType>()) {
      emit(p->type);
    } else {
      auto type_str = type->to_string();
      emit(type_str);
    }
  }

  void emit(IRNode *ir) {
    TI_ASSERT(ir);
    ir->accept(this);
  }

  void emit(const Expr &expr) {
    if (expr) {
      emit(expr.const_value);
      emit(expr.atomic);
      auto *e = expr.expr.get();
      emit(e->stmt);
      emit(e->attributes);
      emit(e->ret_type);
      expr.expr->accept(this);
    } else {
      emit(ExprOpCode::NIL);
    }
  }

  void emit(Stmt *stmt) {
    if (stmt) {
      emit(stmt->get_operands());
      emit(stmt->erased);
      emit(stmt->fields_registered);
      emit(stmt->ret_type);
      stmt->accept(this);
    } else {
      emit(StmtOpCode::NIL);
    }
  }

  void emit(std::size_t size) {
    emit_pod(size);
  }

  void emit(std::uint8_t u8) {
    emit_pod(u8);
  }

  void emit(int i) {
    emit_pod(i);
  }

  void emit(bool v) {
    emit_pod(v);
  }

  void emit(const MemoryAccessOptions &mem_access_options) {
    auto all_options = mem_access_options.get_all();
    emit(static_cast<std::size_t>(all_options.size()));
    for (const auto &[snode, options] : all_options) {
      emit(snode);
      emit(static_cast<std::size_t>(options.size()));
      for (auto e : options) {
        emit(e);
      }
    }
  }

#define DEFINE_EMIT_ENUM(EnumType) \
  void emit(EnumType type) {       \
    emit_pod(type);                \
  }

  DEFINE_EMIT_ENUM(ExprOpCode);
  DEFINE_EMIT_ENUM(StmtOpCode);
  DEFINE_EMIT_ENUM(PrimitiveTypeID);
  DEFINE_EMIT_ENUM(UnaryOpType);
  DEFINE_EMIT_ENUM(BinaryOpType);
  DEFINE_EMIT_ENUM(TernaryOpType);
  DEFINE_EMIT_ENUM(AtomicOpType);
  DEFINE_EMIT_ENUM(SNodeOpType);
  DEFINE_EMIT_ENUM(ForLoopType);
  DEFINE_EMIT_ENUM(SNodeAccessFlag);
  DEFINE_EMIT_ENUM(MeshRelationAccessType);
  DEFINE_EMIT_ENUM(ExternalFuncType);
  DEFINE_EMIT_ENUM(TextureOpType);
  DEFINE_EMIT_ENUM(mesh::MeshElementType);
  DEFINE_EMIT_ENUM(mesh::MeshRelationType);
  DEFINE_EMIT_ENUM(mesh::ConvType);
  DEFINE_EMIT_ENUM(SNodeGradType);

#undef DEFINE_EMIT_ENUM

  Program *prog_{nullptr};
  std::ostream *os_{nullptr};
  std::unordered_set<SNode *> snode_tree_roots_;
  std::unordered_map<Function *, std::size_t> real_funcs_;
  std::vector<char> string_pool_;
};

}  // namespace

void gen_offline_cache_key(Program *prog, IRNode *ast, std::ostream *os) {
  ASTSerializer::run(prog, ast, os);
}

}  // namespace lang
}  // namespace taichi
