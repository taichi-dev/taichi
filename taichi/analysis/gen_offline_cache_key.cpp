#include <unordered_map>
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/common/logging.h"
#include "taichi/ir/expr.h"
#include "taichi/ir/expression_printer.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/type.h"
#include "taichi/program/function.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

namespace {

enum class StmtOpCode : std::uint8_t {
  EnterBlock,
  ExitBlock,
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

class ASTSerializer : public IRVisitor {
 public:
  ASTSerializer(Program *prog, ExpressionPrinter *expr_printer, std::ostream *os)
   : prog_(prog), os_(os), expr_printer_(expr_printer) {
    this->allow_undefined_visitor = true;
    expr_printer_->set_ostream(os);
  }

  void set_ostream(std::ostream *os) {
    this->os_ = os;
  }

  std::ostream *get_ostream() {
    return this->os_;
  }

  void visit(Block *block) override {
    emit(StmtOpCode::EnterBlock);
    emit(static_cast<std::size_t>(block->statements.size()));
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
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
    emit(stmt->ret_type);
    emit(stmt->ident);
  }

  void visit(FrontendAssertStmt *stmt) override {
    emit(StmtOpCode::FrontendAssertStmt);
    emit(stmt->cond);
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    emit(StmtOpCode::FrontendSNodeOpStmt);
    emit(stmt->op_type);
    emit(stmt->snode);
    std::size_t count = stmt->indices.size();
    if (stmt->val.expr) ++count;
    emit(count);
    for (const auto &i : stmt->indices.exprs) {
      emit(i);
    }
    if (stmt->val.expr) {
      emit(stmt->val);
    }
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
        const auto &str = std::get<std::string>(c);
        emit(str);
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
      if (stmt->global_var.is<GlobalVariableExpression>()) {
        emit(stmt->global_var.cast<GlobalVariableExpression>()->snode);
      } else {
        emit(stmt->global_var);
      }
    }
    emit(stmt->bit_vectorize);
    emit(stmt->num_cpu_threads);
    emit(stmt->strictly_serialized);
    emit(stmt->mem_access_opt);
    emit(stmt->block_dim);
    emit(stmt->body.get());
  }

  void visit(FrontendReturnStmt *stmt) override {
    emit(StmtOpCode::FrontendReturnStmt);
    emit(stmt->ret_type);
    emit(stmt->values.exprs);
  }

  void visit(FrontendExternalFuncStmt *stmt) override {
    // Note: The result of serializing FrontendExternalFuncStmt is not parsable now
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
    // Temporary: using ExpressionOfflineCacheKeyGenerator, which will be refactored
    ExpressionOfflineCacheKeyGenerator generator(prog);
    ASTSerializer serializer(prog, &generator, os);
    ast->accept(&serializer);
    serializer.emit_dependencies();
  }

 private:
  void emit_dependencies() {
    // Serialize dependent real-func recursively
    std::ostringstream temp_oss;
    auto *curr_os = this->get_ostream();
    this->set_ostream(&temp_oss);
    expr_printer_->set_ostream(&temp_oss);
    std::size_t last_size{0};
    do {
      last_size = real_funcs_.size();
      for (auto &[func, visited] : real_funcs_) {
        if (!visited) {
          visited = true;
          func->ir->accept(this); // Maybe add new func
        }
      }
    } while(real_funcs_.size() > last_size);
    this->set_ostream(curr_os);
    expr_printer_->set_ostream(curr_os);
    emit(static_cast<std::size_t>(real_funcs_.size()));
    emit(&temp_oss);

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

  template<typename T>
  void emit_pod(const T &val) {
    static_assert(std::is_pod<T>::value);
    TI_ASSERT(os_);
    os_->write((const char *)&val, sizeof(T));
  }

  void emit_bytes(const char *bytes, std::size_t len) {
    TI_ASSERT(os_);
    os_->write(bytes, len);
  }

  template<typename K, typename V>
  void emit(const std::unordered_map<K, V> &map) {
    emit(static_cast<std::size_t>(map.size()));
    for (const auto &[k, v] : map) {
      emit(k);
      emit(v);
    }
  }

  template<typename T1, typename T2>
  void emit(const std::pair<T1, T2> &pair) {
    emit(pair.first);
    emit(pair.second);
  }

  template<typename K, typename V>
  void emit(const std::map<K, V> &map) {
    emit(static_cast<std::size_t>(map.size()));
    for (const auto &[k, v] : map) {
      emit(k);
      emit(v);
    }
  }

  void emit(std::ostream *os) {
    TI_ASSERT(os_ && os);
    *os_ << os->rdbuf();
  }

  void emit(const std::string &str) {
    std::size_t size = str.size();
    std::size_t offset = string_pool_.size();
    string_pool_.insert(string_pool_.end(), str.begin(), str.end());
    emit(size);
    emit(offset);
  }

  void emit(SNodeOpType type) {
    emit_pod(type);
  }

  void emit(SNode *snode) {
    TI_ASSERT(snode);
    TI_ASSERT(prog_);
    emit(static_cast<std::size_t>(snode->get_snode_tree_id()));
    emit(static_cast<std::size_t>(snode->id));
    auto *root = prog_->get_snode_root(snode->get_snode_tree_id());
    snode_tree_roots_.insert(root);
  }

  void emit(mesh::MeshElementType type) {
    emit_pod(type);
  }

  void emit(mesh::MeshRelationType type) {
    emit_pod(type);
  }

  void emit(mesh::ConvType type) {
    emit_pod(type);
  }

  void emit(const mesh::MeshLocalRelation &r) {
    emit(r.fixed);
    emit(r.value);
    emit(r.patch_offset);
    emit(r.offset);
  }

  void emit(mesh::Mesh *mesh) {
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

  void emit(const std::vector<Identifier> &identifiers) {
    emit(static_cast<std::size_t>(identifiers.size()));
    for (const auto &id : identifiers) {
      emit(id);
    }
  }

  void emit(PrimitiveTypeID type_id) {
    emit_pod(type_id);
  }

  void emit(const DataType &type) {
    if (auto *p = type->cast<PrimitiveType>()) {
      emit(p->type);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void emit(StmtOpCode code) {
    emit_pod(code);
  }

  void emit(IRNode *ir) {
    TI_ASSERT(ir);
    ir->accept(this);
  }

  void emit(const Expr &expr) {
    TI_ASSERT(expr_printer_);
    expr.expr->accept(expr_printer_);
  }

  void emit(const std::vector<Expr> &exprs) {
    emit(static_cast<std::size_t>(exprs.size()));
    for (const auto &e : exprs) {
      emit(e);
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

  void emit(ForLoopType type) {
    emit_pod(type);
  }

  void emit(SNodeAccessFlag flag) {
    emit_pod(flag);
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

  void emit(ExternalFuncType type) {
    emit_pod(type);
  }

  Program *prog_{nullptr};
  std::ostream *os_{nullptr};
  ExpressionPrinter *expr_printer_{nullptr};
  std::unordered_set<SNode*> snode_tree_roots_;
  std::unordered_map<Function*, bool> real_funcs_;
  std::vector<char> string_pool_;
};

}  // namespace

void gen_offline_cache_key(Program *prog, IRNode *ast, std::ostream *os) {
  ASTSerializer::run(prog, ast, os);
}

}  // namespace lang
}  // namespace taichi
