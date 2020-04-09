#include <set>
#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
namespace {

using StmtToOffsetMap = decltype(OffloadedResult::local_to_global_offset);

// Break kernel into multiple parts and emit struct for listgens
class Offloader {
 public:
  Offloader(IRNode *root) {
    run(root);
  }

  void fix_loop_index_load(Stmt *s,
                           Stmt *loop_var,
                           int index,
                           bool is_struct_for) {
    replace_statements_with(
        s,
        [&](Stmt *load) {
          if (auto local_load = load->cast<LocalLoadStmt>()) {
            return local_load->width() == 1 &&
                   local_load->ptr[0].var == loop_var &&
                   local_load->ptr[0].offset == 0;
          }
          return false;
        },
        [&]() { return Stmt::make<LoopIndexStmt>(index, is_struct_for); });
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
          offloaded->begin_stmt = s->begin;
        }
        if (auto val = s->end->cast<ConstStmt>()) {
          offloaded->const_end = true;
          offloaded->end_value = val->val[0].val_int32();
        } else {
          offloaded->end_stmt = s->end;
        }
        offloaded->block_dim = s->block_dim;
        offloaded->num_cpu_threads = s->parallelize;
        fix_loop_index_load(s, s->loop_var, 0, /*is_struct_for=*/false);
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

    for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
      fix_loop_index_load(for_stmt, for_stmt->loop_vars[i],
                          leaf->physical_index_position[i],
                          /*is_struct_for=*/true);
    }

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

/*
After offloading, some local variables/instructions are accessed across
offloaded blocks. This pass promote these local values into global variables.

Steps:
  1. (IdentifyLocalVars) Traverse offloaded blocks to identify out-of-block local LD/ST, instruction
references
  2. Replace alloca with global var initialization (set to 0)
     Replace local LD/ST with global LD/ST
*/

class IdentifyLocalVars : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  // Local variables to global temporary offsets (in bytes)
  StmtToOffsetMap local_to_global;
  // Local variables alloc to its containing offloaded statement
  std::map<Stmt *, Stmt *> inst_to_offloaded;

  Stmt *current_offloaded;
  std::size_t global_offset;

  std::size_t allocate_global(VectorType type) {
    TI_ASSERT(type.width == 1);
    auto ret = global_offset;
    global_offset += data_type_size(type.data_type);
    TI_ASSERT(global_offset < taichi_global_tmp_buffer_size);
    return ret;
  }

  IdentifyLocalVars() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_offloaded = nullptr;
    global_offset = 0;
  }

  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (stmt->begin_stmt) {
      test_and_allocate(stmt->begin_stmt);
    }
    if (stmt->end_stmt) {
      test_and_allocate(stmt->end_stmt);
    }
    if (stmt->body)
      stmt->body->accept(this);
    current_offloaded = nullptr;
  }

  void visit(AllocaStmt *stmt) override {
    TI_ASSERT(current_offloaded);
    inst_to_offloaded[stmt] = current_offloaded;
  }

  void test_and_allocate(Stmt *stmt) {
    if (stmt == nullptr)
      return;
    if (inst_to_offloaded[stmt] == current_offloaded)
      return;
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
    if (current_offloaded != nullptr) {
      // inside a offloaded stmt, record its belong offloaded_stmt
      inst_to_offloaded[stmt] = current_offloaded;
    }
    int n_op = stmt->num_operands();
    for (int i = 0; i < n_op; i++) {
      auto op = stmt->operand(i);
      test_and_allocate(op);
    }
  }

  static OffloadedResult run(IRNode *root) {
    IdentifyLocalVars pass;
    root->accept(&pass);
    OffloadedResult result;
    result.total_size = pass.global_offset;
    result.local_to_global_offset = std::move(pass.local_to_global);
    return result;
  }
};

// Store intermediate values to globals so that statements in later offloaded statement can load
class PromoteIntermediate : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  StmtToOffsetMap local_to_global_offset;
  std::set<Stmt *> stored_to_global;

  explicit PromoteIntermediate(const StmtToOffsetMap &local_to_global_offset)
      : local_to_global_offset(local_to_global_offset) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

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
    PromoteIntermediate pass(local_to_global_offset);
    while (true) {
      try {
        root->accept(&pass);
      } catch (IRModified) {
        continue;
      }
      break;
    }
  }
};

class InstToOffload : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  // Local variables to its containing offloaded statement
  std::map<Stmt *, Stmt *> inst_to_offloaded;

  Stmt *current_offloaded;

  InstToOffload() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    current_offloaded = nullptr;
  }

  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (stmt->body)
      stmt->body->accept(this);
    current_offloaded = nullptr;
  }

  void visit(Stmt *stmt) override {
    if (current_offloaded != nullptr) {
      // inside a offloaded stmt, record its belonging offloaded_stmt
      inst_to_offloaded[stmt] = current_offloaded;
    }
  }

  static std::map<Stmt *, Stmt *> run(IRNode *ir) {
    InstToOffload pass;
    ir->accept(&pass);
    return pass.inst_to_offloaded;
  }
};

class PromoteLocals : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  StmtToOffsetMap local_to_global_offset;
  std::map<Stmt *, VectorType> local_to_global_vector_type;
  std::map<Stmt *, Stmt *> inst_to_offload;

  explicit PromoteLocals(const StmtToOffsetMap &local_to_global_offset,
                         std::map<Stmt *, Stmt *> inst_to_offload)
      : local_to_global_offset(local_to_global_offset),
        inst_to_offload(inst_to_offload) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->body)
      stmt->body->accept(this);
    if (stmt->task_type == OffloadedStmt::TaskType::range_for) {
      if (!stmt->const_begin)
        stmt->begin_offset = local_to_global_offset[stmt->begin_stmt];
      if (!stmt->const_end)
        stmt->end_offset = local_to_global_offset[stmt->end_stmt];
    }
  }

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

  // Generic visitor
  // TODO: maybe other visitors do the same thing as this?
  void visit(Stmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    int n_op = stmt->num_operands();
    bool modified = false;
    for (int i = 0; i < n_op; i++) {
      auto op = stmt->operand(i);
      if (op == nullptr)
        continue;
      if (local_to_global_offset.find(op) == local_to_global_offset.end())
        continue;
      if (inst_to_offload[stmt] == inst_to_offload[op])  // same OffloadedStmt
        continue;

      auto global = Stmt::make<GlobalTemporaryStmt>(local_to_global_offset[op],
                                                    op->ret_type);
      auto load = Stmt::make<GlobalLoadStmt>(global.get());
      stmt->set_operand(i, load.get());
      stmt->insert_before_me(std::move(global));
      stmt->insert_before_me(std::move(load));
      modified = true;
    }
    if (modified)
      throw IRModified();
  }

  static void run(IRNode *root,
                  std::map<Stmt *, Stmt *> inst_to_offload,
                  const StmtToOffsetMap &local_to_global_offset) {
    PromoteLocals pass(local_to_global_offset, inst_to_offload);
    while (true) {
      try {
        root->accept(&pass);
      } catch (IRModified) {
        continue;
      }
      break;
    }
  }
};

void insert_gc(IRNode *root) {
  auto *b = dynamic_cast<Block *>(root);
  TI_ASSERT(b);
  std::vector<std::pair<int, std::vector<SNode *>>> gc_statements;
  for (int i = 0; i < (int)b->statements.size(); i++) {
    auto snodes = irpass::gather_deactivations(b->statements[i].get());
    gc_statements.emplace_back(
        std::make_pair(i, std::vector<SNode *>(snodes.begin(), snodes.end())));
  }

  for (int i = (int)b->statements.size() - 1; i >= 0; i--) {
    auto snodes = gc_statements[i].second;
    for (auto j = 0; j < snodes.size(); j++) {
      b->statements.insert(
          b->statements.begin() + i + 1,
          Stmt::make<OffloadedStmt>(OffloadedStmt::TaskType::gc, snodes[j]));
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
  irpass::typecheck(root);
  irpass::fix_block_parents(root);
  {
    result = IdentifyLocalVars::run(root);
    fix_block_parents(root);
    PromoteIntermediate::run(root, result.local_to_global_offset);
    fix_block_parents(root);
    auto inst_to_offload = InstToOffload::run(root);
    PromoteLocals::run(root, inst_to_offload, result.local_to_global_offset);
    fix_block_parents(root);
  }
  insert_gc(root);
  // TODO(k-ye): Move this into its own pass. However, we need to wait for all
  // backends to integrate with https://github.com/taichi-dev/taichi/pull/700
  AssociateContinueScope::run(root);
  typecheck(root);
  re_id(root);
  return result;
}

}  // namespace irpass

TLANG_NAMESPACE_END
