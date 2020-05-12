#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include <set>
#include <unordered_map>
#include <utility>

TLANG_NAMESPACE_BEGIN

namespace irpass {
namespace {

using StmtToOffsetMap = decltype(OffloadedResult::local_to_global_offset);

std::unique_ptr<std::unordered_map<OffloadedStmt *, Stmt *>> begin_stmt,
    end_stmt;

// Break kernel into multiple parts and emit struct for listgens
class Offloader {
 public:
  Offloader(IRNode *root) {
    begin_stmt =
        std::make_unique<std::unordered_map<OffloadedStmt *, Stmt *>>();
    end_stmt = std::make_unique<std::unordered_map<OffloadedStmt *, Stmt *>>();
    run(root);
  }

  void run(IRNode *root) {
    auto root_block = dynamic_cast<Block *>(root);
    auto root_statements = std::move(root_block->statements);
    root_block->statements.clear();

    auto pending_serial_statements =
        Stmt::make_typed<OffloadedStmt>(OffloadedStmt::TaskType::serial);

    auto assemble_serial_statements = [&]() {
      if (!pending_serial_statements->body->statements.empty()) {
        root_block->insert(std::move(pending_serial_statements));
        pending_serial_statements =
            Stmt::make_typed<OffloadedStmt>(OffloadedStmt::TaskType::serial);
      }
    };

    for (int i = 0; i < (int)root_statements.size(); i++) {
      auto &stmt = root_statements[i];
      if (auto s = stmt->cast<RangeForStmt>(); s && !s->strictly_serialized) {
        assemble_serial_statements();
        auto offloaded =
            Stmt::make_typed<OffloadedStmt>(OffloadedStmt::TaskType::range_for);
        offloaded->body = std::make_unique<Block>();
        if (auto val = s->begin->cast<ConstStmt>()) {
          offloaded->const_begin = true;
          offloaded->begin_value = val->val[0].val_int32();
        } else {
          begin_stmt->insert(std::make_pair(offloaded.get(), s->begin));
        }
        if (auto val = s->end->cast<ConstStmt>()) {
          offloaded->const_end = true;
          offloaded->end_value = val->val[0].val_int32();
        } else {
          end_stmt->insert(std::make_pair(offloaded.get(), s->end));
        }
        offloaded->block_dim = s->block_dim;
        offloaded->num_cpu_threads = s->parallelize;
        replace_all_usages_with(s, s, offloaded.get());
        for (int j = 0; j < (int)s->body->statements.size(); j++) {
          offloaded->body->insert(std::move(s->body->statements[j]));
        }
        root_block->insert(std::move(offloaded));
      } else if (auto s = stmt->cast<StructForStmt>()) {
        assemble_serial_statements();
        emit_struct_for(s, root_block);
      } else {
        pending_serial_statements->body->insert(std::move(stmt));
      }
    }
    assemble_serial_statements();
  }

  void emit_struct_for(StructForStmt *for_stmt, Block *root_block) {
    auto leaf = for_stmt->snode;
    // make a list of nodes, from the leaf block (instead of 'place') to root
    std::vector<SNode *> path;
    // leaf is the place (scalar)
    // leaf->parent is the leaf block
    // so listgen should be invoked from the root to leaf->parent
    for (auto p = leaf; p; p = p->parent) {
      path.push_back(p);
    }
    std::reverse(path.begin(), path.end());

    for (int i = 1; i < path.size(); i++) {
      auto snode_child = path[i];
      auto offloaded_clear_list =
          Stmt::make_typed<OffloadedStmt>(OffloadedStmt::TaskType::clear_list);
      offloaded_clear_list->snode = snode_child;
      root_block->insert(std::move(offloaded_clear_list));
      auto offloaded_listgen =
          Stmt::make_typed<OffloadedStmt>(OffloadedStmt::TaskType::listgen);
      offloaded_listgen->snode = snode_child;
      root_block->insert(std::move(offloaded_listgen));
    }

    auto offloaded_struct_for =
        Stmt::make_typed<OffloadedStmt>(OffloadedStmt::TaskType::struct_for);

    replace_all_usages_with(for_stmt, for_stmt, offloaded_struct_for.get());

    for (int i = 0; i < (int)for_stmt->body->statements.size(); i++) {
      offloaded_struct_for->body->insert(
          std::move(for_stmt->body->statements[i]));
    }

    offloaded_struct_for->block_dim = for_stmt->block_dim;
    offloaded_struct_for->snode = for_stmt->snode;
    offloaded_struct_for->num_cpu_threads = for_stmt->parallelize;

    root_block->insert(std::move(offloaded_struct_for));
  }
};

// Build a mapping from all statements to its containing OffloadedStmt
class StmtToOffloaded : public BasicStmtVisitor {
 private:
  StmtToOffloaded() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_offloaded = nullptr;
  }

 public:
  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    stmt_to_offloaded[stmt] = current_offloaded;
    if (stmt->body)
      stmt->body->accept(this);
    current_offloaded = nullptr;
  }

  void visit(Stmt *stmt) override {
    if (current_offloaded != nullptr) {
      // inside a offloaded stmt, record its belonging offloaded_stmt
      stmt_to_offloaded[stmt] = current_offloaded;
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (current_offloaded != nullptr) {
      // inside a offloaded stmt, record its belonging offloaded_stmt
      stmt_to_offloaded[stmt] = current_offloaded;
    }
  }

 public:
  static std::unordered_map<Stmt *, Stmt *> run(IRNode *ir) {
    StmtToOffloaded pass;
    ir->accept(&pass);
    return pass.stmt_to_offloaded;
  }

 private:
  using BasicStmtVisitor::visit;

  // Local variables to its containing offloaded statement
  std::unordered_map<Stmt *, Stmt *> stmt_to_offloaded;

  Stmt *current_offloaded;
};

/*
After offloading, some local variables/instructions are accessed across
offloaded blocks. This pass promote these local values into global variables.

Steps:
  1. IdentifyValuesUsedInOtherOffloads
  2. PromoteIntermediateToGlobalTmp
  3. FixCrossOffloadReferences
*/

// Traverse offloaded blocks to identify out-of-offload local LD/ST and
// statement references
class IdentifyValuesUsedInOtherOffloads : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 private:
  IdentifyValuesUsedInOtherOffloads(
      const std::unordered_map<Stmt *, Stmt *> &stmt_to_offloaded)
      : stmt_to_offloaded(stmt_to_offloaded) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_offloaded = nullptr;
    global_offset = 0;
  }

  std::size_t allocate_global(VectorType type) {
    TI_ASSERT(type.width == 1);
    auto ret = global_offset;
    global_offset += data_type_size(type.data_type);
    TI_ASSERT(global_offset < taichi_global_tmp_buffer_size);
    return ret;
  }

 public:
  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (auto begin = begin_stmt->find(stmt); begin != begin_stmt->end()) {
      test_and_allocate(begin->second);
    }
    if (auto end = end_stmt->find(stmt); end != end_stmt->end()) {
      test_and_allocate(end->second);
    }
    if (stmt->body)
      stmt->body->accept(this);
    current_offloaded = nullptr;
  }

  void visit(AllocaStmt *stmt) override {
    TI_ASSERT(current_offloaded);
  }

  void test_and_allocate(Stmt *stmt) {
    if (stmt == nullptr)
      return;
    if (stmt_to_offloaded[stmt] == current_offloaded)
      return;
    if (advanced_optimization) {
      if (stmt->is<ConstStmt>()) {
        // Directly insert copies of ConstStmts later
        return;
      }
    }
    if (local_to_global.find(stmt) == local_to_global.end()) {
      // Not yet allocated
      local_to_global[stmt] = allocate_global(stmt->ret_type);
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(current_offloaded);
    TI_ASSERT(stmt->width() == 1);
    test_and_allocate(stmt->ptr[0].var);
  }

  void visit(LocalStoreStmt *stmt) override {
    TI_ASSERT(current_offloaded);
    TI_ASSERT(stmt->width() == 1);
    test_and_allocate(stmt->ptr);
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(current_offloaded);
    TI_ASSERT(stmt->width() == 1);
    if (stmt->dest->is<AllocaStmt>()) {
      test_and_allocate(stmt->dest);
    }
  }

  void visit(Stmt *stmt) override {
    int n_op = stmt->num_operands();
    for (int i = 0; i < n_op; i++) {
      auto op = stmt->operand(i);
      test_and_allocate(op);
    }
  }

  static OffloadedResult run(
      IRNode *root,
      const std::unordered_map<Stmt *, Stmt *> &stmt_to_offloaded) {
    IdentifyValuesUsedInOtherOffloads pass(stmt_to_offloaded);
    root->accept(&pass);
    OffloadedResult result;
    result.total_size = pass.global_offset;
    result.local_to_global_offset = std::move(pass.local_to_global);
    return result;
  }

 private:
  // Local variables to global temporary offsets (in bytes)
  StmtToOffsetMap local_to_global;
  std::unordered_map<Stmt *, Stmt *> stmt_to_offloaded;
  Stmt *current_offloaded;
  std::size_t global_offset;
};

// Store intermediate values to globals so that statements in later offloaded
// statement can load
class PromoteIntermediateToGlobalTmp : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 private:
  explicit PromoteIntermediateToGlobalTmp(
      const StmtToOffsetMap &local_to_global_offset)
      : local_to_global_offset(local_to_global_offset) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

 public:
  void visit(Stmt *stmt) override {
    if (!stmt->is<AllocaStmt>() &&
        local_to_global_offset.find(stmt) != local_to_global_offset.end() &&
        stored_to_global.find(stmt) == stored_to_global.end()) {
      stored_to_global.insert(stmt);
      auto offset = local_to_global_offset[stmt];
      auto ptr = stmt->insert_after_me(
          Stmt::make<GlobalTemporaryStmt>(offset, stmt->ret_type));
      ptr->insert_after_me(Stmt::make<GlobalStoreStmt>(ptr, stmt));
      throw IRModified();
    }
  }

  static void run(IRNode *root, const StmtToOffsetMap &local_to_global_offset) {
    PromoteIntermediateToGlobalTmp pass(local_to_global_offset);
    while (true) {
      try {
        root->accept(&pass);
      } catch (IRModified) {
        continue;
      }
      break;
    }
  }

 private:
  StmtToOffsetMap local_to_global_offset;
  std::set<Stmt *> stored_to_global;
};

class FixCrossOffloadReferences : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 private:
  FixCrossOffloadReferences(
      const StmtToOffsetMap &local_to_global_offset,
      std::unordered_map<Stmt *, Stmt *> stmt_to_offloaded)
      : local_to_global_offset(local_to_global_offset),
        stmt_to_offloaded(std::move(stmt_to_offloaded)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->body)
      stmt->body->accept(this);
    if (stmt->task_type == OffloadedStmt::TaskType::range_for) {
      if (!stmt->const_begin)
        stmt->begin_offset =
            local_to_global_offset[begin_stmt->find(stmt)->second];
      if (!stmt->const_end)
        stmt->end_offset = local_to_global_offset[end_stmt->find(stmt)->second];
    }
  }

  // Replace alloca with global var initialization (set to 0)
  void visit(AllocaStmt *stmt) override {
    if (local_to_global_offset.find(stmt) == local_to_global_offset.end())
      return;
    VecStatement replacement;
    auto ret_type = stmt->ret_type;
    local_to_global_vector_type[stmt] = ret_type;
    auto ptr = replacement.push_back<GlobalTemporaryStmt>(
        local_to_global_offset[stmt], ret_type);
    LaneAttribute<TypedConstant> zeros(std::vector<TypedConstant>(
        stmt->width(), TypedConstant(stmt->ret_type.data_type)));
    auto const_zeros = replacement.push_back<ConstStmt>(zeros);
    replacement.push_back<GlobalStoreStmt>(ptr, const_zeros);

    stmt->parent->replace_with(stmt, std::move(replacement), false);
    throw IRModified();
  }

  // Replace local LD/ST with global LD/ST
  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto alloca = stmt->ptr[0].var;
    if (local_to_global_offset.find(alloca) == local_to_global_offset.end())
      return;

    VecStatement replacement;
    auto ret_type = stmt->ret_type;

    auto ptr = replacement.push_back<GlobalTemporaryStmt>(
        local_to_global_offset[alloca], ret_type);
    replacement.push_back<GlobalLoadStmt>(ptr);

    stmt->parent->replace_with(stmt, std::move(replacement));
    throw IRModified();
  }

  void visit(LocalStoreStmt *stmt) override {
    if (visit_operand(stmt, stmt->locate_operand(&stmt->data)))
      throw IRModified();
    TI_ASSERT(stmt->width() == 1);
    auto alloca = stmt->ptr;
    if (local_to_global_offset.find(alloca) == local_to_global_offset.end())
      return;

    VecStatement replacement;
    auto ret_type = stmt->ret_type;

    auto ptr = replacement.push_back<GlobalTemporaryStmt>(
        local_to_global_offset[alloca], ret_type);
    replacement.push_back<GlobalStoreStmt>(ptr, stmt->data);

    stmt->parent->replace_with(stmt, std::move(replacement));
    throw IRModified();
  }

  void visit(AtomicOpStmt *stmt) override {
    if (visit_operand(stmt, stmt->locate_operand(&stmt->val)))
      throw IRModified();
    TI_ASSERT(stmt->width() == 1);
    auto alloca = stmt->dest;
    if (local_to_global_offset.find(alloca) == local_to_global_offset.end())
      return;

    VecStatement replacement;
    auto ret_type = stmt->dest->ret_type;

    auto ptr = replacement.push_back<GlobalTemporaryStmt>(
        local_to_global_offset[alloca], ret_type);
    replacement.push_back<AtomicOpStmt>(stmt->op_type, ptr, stmt->val);

    stmt->parent->replace_with(stmt, std::move(replacement));
    throw IRModified();
  }

  bool visit_operand(Stmt *stmt, int index) {
    // return true if modified
    TI_ASSERT(index >= 0 && index < stmt->num_operands());
    auto op = stmt->operand(index);
    if (op == nullptr)
      return false;
    if (stmt_to_offloaded[stmt] == stmt_to_offloaded[op])  // same OffloadedStmt
      return false;
    if (advanced_optimization) {
      if (op->is<ConstStmt>()) {
        auto copy = op->as<ConstStmt>()->copy();
        stmt_to_offloaded[copy.get()] = stmt_to_offloaded[stmt];
        stmt->set_operand(index, copy.get());
        stmt->insert_before_me(std::move(copy));
        return true;
      }
    }
    if (local_to_global_offset.find(op) == local_to_global_offset.end())
      return false;

    auto global = Stmt::make<GlobalTemporaryStmt>(local_to_global_offset[op],
                                                  op->ret_type);
    auto load = Stmt::make<GlobalLoadStmt>(global.get());
    stmt_to_offloaded[load.get()] = stmt_to_offloaded[stmt];
    stmt->set_operand(index, load.get());
    stmt->insert_before_me(std::move(global));
    stmt->insert_before_me(std::move(load));
    return true;
  }

  // Generic visitor
  void visit(Stmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    int n_op = stmt->num_operands();
    bool modified = false;
    for (int i = 0; i < n_op; i++) {
      if (visit_operand(stmt, i))
        modified = true;
    }
    if (modified)
      throw IRModified();
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    int n_op = stmt->num_operands();
    bool modified = false;
    for (int i = 0; i < n_op; i++) {
      if (visit_operand(stmt, i))
        modified = true;
    }
    if (modified)
      throw IRModified();
  }

 public:
  static void run(IRNode *root,
                  std::unordered_map<Stmt *, Stmt *> stmt_to_offloaded,
                  const StmtToOffsetMap &local_to_global_offset) {
    FixCrossOffloadReferences pass(local_to_global_offset, stmt_to_offloaded);
    while (true) {
      try {
        root->accept(&pass);
      } catch (IRModified) {
        continue;
      }
      break;
    }
  }

 private:
  StmtToOffsetMap local_to_global_offset;
  std::unordered_map<Stmt *, VectorType> local_to_global_vector_type;
  std::unordered_map<Stmt *, Stmt *> stmt_to_offloaded;
};

void insert_gc(IRNode *root) {
  auto *b = dynamic_cast<Block *>(root);
  TI_ASSERT(b);
  std::vector<std::pair<int, std::vector<SNode *>>> gc_statements;
  for (int i = 0; i < (int)b->statements.size(); i++) {
    auto snodes =
        irpass::analysis::gather_deactivations(b->statements[i].get());
    gc_statements.emplace_back(
        std::make_pair(i, std::vector<SNode *>(snodes.begin(), snodes.end())));
  }

  for (int i = (int)b->statements.size() - 1; i >= 0; i--) {
    auto snodes = gc_statements[i].second;
    for (auto *snode : snodes) {
      if (is_gc_able(snode->type)) {
        b->insert(Stmt::make<OffloadedStmt>(OffloadedStmt::TaskType::gc, snode),
                  i + 1);
      }
    }
  }
}

class AssociateContinueScope : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  using Parent = BasicStmtVisitor;

  void visit(WhileStmt *stmt) override {
    auto *old_loop = cur_internal_loop_;
    cur_internal_loop_ = stmt;
    Parent::visit(stmt);
    cur_internal_loop_ = old_loop;
  }

  void visit(RangeForStmt *stmt) override {
    auto *old_loop = cur_internal_loop_;
    cur_internal_loop_ = stmt;
    Parent::visit(stmt);
    cur_internal_loop_ = old_loop;
  }

  void visit(StructForStmt *stmt) override {
    TI_ERROR("struct_for cannot be nested inside a kernel, stmt={}",
             stmt->name());
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(cur_offloaded_stmt_ == nullptr);
    TI_ASSERT(cur_internal_loop_ == nullptr);
    cur_offloaded_stmt_ = stmt;
    Parent::visit(stmt);
    cur_offloaded_stmt_ = nullptr;
  }

  void visit(ContinueStmt *stmt) override {
    if (stmt->scope == nullptr) {
      if (cur_internal_loop_ != nullptr) {
        stmt->scope = cur_internal_loop_;
      } else {
        stmt->scope = cur_offloaded_stmt_;
      }
      modified_ = true;
    }
    TI_ASSERT(stmt->scope != nullptr);
  }

  static void run(IRNode *root) {
    while (true) {
      AssociateContinueScope pass;
      root->accept(&pass);
      if (!pass.modified_) {
        break;
      }
    }
  }

 private:
  explicit AssociateContinueScope()
      : modified_(false),
        cur_offloaded_stmt_(nullptr),
        cur_internal_loop_(nullptr) {
  }

  bool modified_;
  OffloadedStmt *cur_offloaded_stmt_;
  Stmt *cur_internal_loop_;
};

}  // namespace

OffloadedResult offload(IRNode *root) {
  OffloadedResult result;
  Offloader _(root);
  typecheck(root);
  fix_block_parents(root);
  {
    auto stmt_to_offloaded = StmtToOffloaded::run(root);
    result = IdentifyValuesUsedInOtherOffloads::run(root, stmt_to_offloaded);
    PromoteIntermediateToGlobalTmp::run(root, result.local_to_global_offset);
    stmt_to_offloaded = StmtToOffloaded::run(root);
    FixCrossOffloadReferences::run(root, stmt_to_offloaded,
                                   result.local_to_global_offset);
    fix_block_parents(root);
  }
  insert_gc(root);
  // TODO(k-ye): Move this into its own pass. However, we need to wait for all
  // backends to integrate with https://github.com/taichi-dev/taichi/pull/700
  AssociateContinueScope::run(root);
  typecheck(root);
  re_id(root);
  fix_block_parents(root);
  return result;
}

}  // namespace irpass

TLANG_NAMESPACE_END
