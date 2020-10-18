// TODO: gradually cppize statements.h
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/bit.h"

TLANG_NAMESPACE_BEGIN

bool ContinueStmt::as_return() const {
  TI_ASSERT(scope != nullptr);
  if (auto *offl = scope->cast<OffloadedStmt>(); offl) {
    TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
              offl->task_type == OffloadedStmt::TaskType::struct_for);
    return true;
  }
  return false;
}

UnaryOpStmt::UnaryOpStmt(UnaryOpType op_type, Stmt *operand)
    : op_type(op_type), operand(operand) {
  TI_ASSERT(!operand->is<AllocaStmt>());
  cast_type = PrimitiveType::unknown;
  TI_STMT_REG_FIELDS;
}

bool UnaryOpStmt::is_cast() const {
  return unary_op_is_cast(op_type);
}

bool UnaryOpStmt::same_operation(UnaryOpStmt *o) const {
  if (op_type == o->op_type) {
    if (is_cast()) {
      return cast_type == o->cast_type;
    } else {
      return true;
    }
  }
  return false;
}

ExternalPtrStmt::ExternalPtrStmt(const LaneAttribute<Stmt *> &base_ptrs,
                                 const std::vector<Stmt *> &indices)
    : base_ptrs(base_ptrs), indices(indices) {
  DataType dt = PrimitiveType::f32;
  for (int i = 0; i < (int)base_ptrs.size(); i++) {
    TI_ASSERT(base_ptrs[i] != nullptr);
    TI_ASSERT(base_ptrs[i]->is<ArgLoadStmt>());
  }
  TI_ASSERT(base_ptrs.size() == 1);
  element_type() = dt;
  TI_STMT_REG_FIELDS;
}

GlobalPtrStmt::GlobalPtrStmt(const LaneAttribute<SNode *> &snodes,
                             const std::vector<Stmt *> &indices,
                             bool activate)
    : snodes(snodes), indices(indices), activate(activate) {
  for (int i = 0; i < (int)snodes.size(); i++) {
    TI_ASSERT(snodes[i] != nullptr);
    TI_ASSERT(snodes[0]->dt == snodes[i]->dt);
  }
  TI_ASSERT(snodes.size() == 1);
  element_type() = snodes[0]->dt;
  TI_STMT_REG_FIELDS;
}

bool GlobalPtrStmt::is_element_wise(SNode *snode) const {
  if (snode == nullptr) {
    // check every SNode when "snode" is nullptr
    for (const auto &snode_i : snodes.data) {
      if (!is_element_wise(snode_i)) {
        return false;
      }
    }
    return true;
  }
  // check if this statement is element-wise on a specific SNode, i.e., argument
  // "snode"
  for (int i = 0; i < (int)indices.size(); i++) {
    if (auto loop_index_i = indices[i]->cast<LoopIndexStmt>();
        !(loop_index_i && loop_index_i->loop->is<OffloadedStmt>() &&
          loop_index_i->index == snode->physical_index_position[i])) {
      return false;
    }
  }
  return true;
}

SNodeOpStmt::SNodeOpStmt(SNodeOpType op_type,
                         SNode *snode,
                         Stmt *ptr,
                         Stmt *val)
    : op_type(op_type), snode(snode), ptr(ptr), val(val) {
  element_type() = PrimitiveType::i32;
  TI_STMT_REG_FIELDS;
}

SNodeOpStmt::SNodeOpStmt(SNodeOpType op_type,
                         SNode *snode,
                         const std::vector<Stmt *> &indices)
    : op_type(op_type), snode(snode), indices(indices) {
  ptr = nullptr;
  val = nullptr;
  TI_ASSERT(op_type == SNodeOpType::is_active ||
            op_type == SNodeOpType::deactivate ||
            op_type == SNodeOpType::activate);
  element_type() = PrimitiveType::i32;
  TI_STMT_REG_FIELDS;
}

bool SNodeOpStmt::activation_related(SNodeOpType op) {
  return op == SNodeOpType::activate || op == SNodeOpType::deactivate ||
         op == SNodeOpType::is_active;
}

bool SNodeOpStmt::need_activation(SNodeOpType op) {
  return op == SNodeOpType::activate || op == SNodeOpType::append;
}

ExternalTensorShapeAlongAxisStmt::ExternalTensorShapeAlongAxisStmt(int axis,
                                                                   int arg_id)
    : axis(axis), arg_id(arg_id) {
  TI_STMT_REG_FIELDS;
}

Stmt *LocalLoadStmt::previous_store_or_alloca_in_block() {
  int position = parent->locate(this);
  // TI_ASSERT(width() == 1);
  // TI_ASSERT(this->ptr[0].offset == 0);
  for (int i = position - 1; i >= 0; i--) {
    if (parent->statements[i]->is<LocalStoreStmt>()) {
      auto store = parent->statements[i]->as<LocalStoreStmt>();
      // TI_ASSERT(store->width() == 1);
      if (store->ptr == this->ptr[0].var) {
        // found
        return store;
      }
    } else if (parent->statements[i]->is<AllocaStmt>()) {
      auto alloca = parent->statements[i]->as<AllocaStmt>();
      // TI_ASSERT(alloca->width() == 1);
      if (alloca == this->ptr[0].var) {
        return alloca;
      }
    }
  }
  return nullptr;
}

bool LocalLoadStmt::same_source() const {
  for (int i = 1; i < (int)ptr.size(); i++) {
    if (ptr[i].var != ptr[0].var)
      return false;
  }
  return true;
}

bool LocalLoadStmt::has_source(Stmt *alloca) const {
  for (int i = 0; i < width(); i++) {
    if (ptr[i].var == alloca)
      return true;
  }
  return false;
}

IfStmt::IfStmt(Stmt *cond)
    : cond(cond), true_mask(nullptr), false_mask(nullptr) {
  TI_STMT_REG_FIELDS;
}

void IfStmt::set_true_statements(std::unique_ptr<Block> &&new_true_statements) {
  true_statements = std::move(new_true_statements);
  if (true_statements)
    true_statements->parent_stmt = this;
}

void IfStmt::set_false_statements(
    std::unique_ptr<Block> &&new_false_statements) {
  false_statements = std::move(new_false_statements);
  if (false_statements)
    false_statements->parent_stmt = this;
}

std::unique_ptr<Stmt> IfStmt::clone() const {
  auto new_stmt = std::make_unique<IfStmt>(cond);
  new_stmt->true_mask = true_mask;
  new_stmt->false_mask = false_mask;
  if (true_statements)
    new_stmt->set_true_statements(true_statements->clone());
  if (false_statements)
    new_stmt->set_false_statements(false_statements->clone());
  return new_stmt;
}

std::unique_ptr<ConstStmt> ConstStmt::copy() {
  return std::make_unique<ConstStmt>(val);
}

RangeForStmt::RangeForStmt(Stmt *begin,
                           Stmt *end,
                           std::unique_ptr<Block> &&body,
                           int vectorize,
                           int parallelize,
                           int block_dim,
                           bool strictly_serialized)
    : begin(begin),
      end(end),
      body(std::move(body)),
      vectorize(vectorize),
      parallelize(parallelize),
      block_dim(block_dim),
      strictly_serialized(strictly_serialized) {
  reversed = false;
  this->body->parent_stmt = this;
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> RangeForStmt::clone() const {
  auto new_stmt = std::make_unique<RangeForStmt>(
      begin, end, body->clone(), vectorize, parallelize, block_dim,
      strictly_serialized);
  new_stmt->reversed = reversed;
  return new_stmt;
}

StructForStmt::StructForStmt(SNode *snode,
                             std::unique_ptr<Block> &&body,
                             int vectorize,
                             int parallelize,
                             int block_dim)
    : snode(snode),
      body(std::move(body)),
      vectorize(vectorize),
      parallelize(parallelize),
      block_dim(block_dim) {
  this->body->parent_stmt = this;
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> StructForStmt::clone() const {
  auto new_stmt = std::make_unique<StructForStmt>(
      snode, body->clone(), vectorize, parallelize, block_dim);
  new_stmt->scratch_opt = scratch_opt;
  return new_stmt;
}

FuncBodyStmt::FuncBodyStmt(const std::string &funcid,
                           std::unique_ptr<Block> &&body)
    : funcid(funcid), body(std::move(body)) {
  if (this->body)
    this->body->parent_stmt = this;
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> FuncBodyStmt::clone() const {
  return std::make_unique<FuncBodyStmt>(funcid, body->clone());
}

WhileStmt::WhileStmt(std::unique_ptr<Block> &&body)
    : mask(nullptr), body(std::move(body)) {
  this->body->parent_stmt = this;
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> WhileStmt::clone() const {
  auto new_stmt = std::make_unique<WhileStmt>(body->clone());
  new_stmt->mask = mask;
  return new_stmt;
}

GetChStmt::GetChStmt(Stmt *input_ptr, int chid)
    : input_ptr(input_ptr), chid(chid) {
  TI_ASSERT(input_ptr->is<SNodeLookupStmt>());
  input_snode = input_ptr->as<SNodeLookupStmt>()->snode;
  output_snode = input_snode->ch[chid].get();
  TI_STMT_REG_FIELDS;
}

OffloadedStmt::OffloadedStmt(OffloadedStmt::TaskType task_type)
    : OffloadedStmt(task_type, nullptr) {
}

OffloadedStmt::OffloadedStmt(OffloadedStmt::TaskType task_type, SNode *snode)
    : task_type(task_type), snode(snode) {
  num_cpu_threads = 1;
  const_begin = false;
  const_end = false;
  begin_value = 0;
  end_value = 0;
  step = 0;
  reversed = false;
  device = get_current_program().config.arch;
  if (has_body()) {
    body = std::make_unique<Block>();
    body->parent_stmt = this;
  }
  TI_STMT_REG_FIELDS;
}

std::string OffloadedStmt::task_name() const {
  if (task_type == TaskType::serial) {
    return "serial";
  } else if (task_type == TaskType::range_for) {
    return "range_for";
  } else if (task_type == TaskType::struct_for) {
    return "struct_for";
  } else if (task_type == TaskType::listgen) {
    TI_ASSERT(snode);
    return fmt::format("listgen_{}", snode->get_node_type_name_hinted());
  } else if (task_type == TaskType::gc) {
    TI_ASSERT(snode);
    return fmt::format("gc_{}", snode->name);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

// static
std::string OffloadedStmt::task_type_name(TaskType tt) {
  return offloaded_task_type_name(tt);
}

std::unique_ptr<Stmt> OffloadedStmt::clone() const {
  auto new_stmt = std::make_unique<OffloadedStmt>(task_type, snode);
  new_stmt->begin_offset = begin_offset;
  new_stmt->end_offset = end_offset;
  new_stmt->const_begin = const_begin;
  new_stmt->const_end = const_end;
  new_stmt->begin_value = begin_value;
  new_stmt->end_value = end_value;
  new_stmt->step = step;
  new_stmt->grid_dim = grid_dim;
  new_stmt->block_dim = block_dim;
  new_stmt->reversed = reversed;
  new_stmt->num_cpu_threads = num_cpu_threads;
  new_stmt->device = device;
  if (body) {
    new_stmt->body = body->clone();
    new_stmt->body->parent_stmt = new_stmt.get();
  }
  return new_stmt;
}

void OffloadedStmt::all_blocks_accept(IRVisitor *visitor) {
  if (tls_prologue)
    tls_prologue->accept(visitor);
  if (bls_prologue)
    bls_prologue->accept(visitor);
  if (body)
    body->accept(visitor);
  if (bls_epilogue)
    bls_epilogue->accept(visitor);
  if (tls_epilogue)
    tls_epilogue->accept(visitor);
}

bool is_clear_list_task(const OffloadedStmt *stmt) {
  return (stmt->task_type == OffloadedStmt::TaskType::serial) &&
         (stmt->body->size() == 1) && stmt->body->back()->is<ClearListStmt>();
}

ClearListStmt::ClearListStmt(SNode *snode) : snode(snode) {
  TI_STMT_REG_FIELDS;
}

int LoopIndexStmt::max_num_bits() const {
  if (auto range_for = loop->cast<RangeForStmt>()) {
    // Return the max number of bits only if both begin and end are
    // non-negative consts.
    if (!range_for->begin->is<ConstStmt>() || !range_for->end->is<ConstStmt>())
      return -1;
    auto begin = range_for->begin->as<ConstStmt>();
    for (int i = 0; i < (int)begin->val.size(); i++) {
      if (begin->val[i].val_int() < 0)
        return -1;
    }
    auto end = range_for->end->as<ConstStmt>();
    int result = 0;
    for (int i = 0; i < (int)end->val.size(); i++) {
      result = std::max(result, (int)bit::ceil_log2int(end->val[i].val_int()));
    }
    return result;
  } else if (auto struct_for = loop->cast<StructForStmt>()) {
    return struct_for->snode->get_num_bits(index);
  } else if (auto offload = loop->cast<OffloadedStmt>()) {
    if (offload->task_type == OffloadedStmt::TaskType::range_for) {
      if (!offload->const_begin || !offload->const_end)
        return -1;
      if (offload->begin_value < 0)
        return -1;
      return bit::ceil_log2int(offload->end_value);
    } else if (offload->task_type == OffloadedStmt::TaskType::struct_for) {
      return offload->snode->get_num_bits(index);
    } else {
      TI_NOT_IMPLEMENTED
    }
  } else {
    TI_NOT_IMPLEMENTED
  }
}

TLANG_NAMESPACE_END
