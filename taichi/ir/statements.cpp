// TODO: gradually cppize statements.h
#include "statements.h"
#include "taichi/program/program.h"
#include "taichi/util/bit.h"

TLANG_NAMESPACE_BEGIN

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
  if (task_type != TaskType::listgen) {
    body = std::make_unique<Block>();
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
  } else if (task_type == TaskType::clear_list) {
    TI_ASSERT(snode);
    return fmt::format("clear_list_{}", snode->get_node_type_name_hinted());
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
#define REGISTER_NAME(x) \
  { TaskType::x, #x }
  const static std::unordered_map<TaskType, std::string> m = {
      REGISTER_NAME(serial),     REGISTER_NAME(range_for),
      REGISTER_NAME(struct_for), REGISTER_NAME(clear_list),
      REGISTER_NAME(listgen),    REGISTER_NAME(gc),
  };
#undef REGISTER_NAME
  return m.find(tt)->second;
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
  new_stmt->block_dim = block_dim;
  new_stmt->reversed = reversed;
  new_stmt->num_cpu_threads = num_cpu_threads;
  new_stmt->device = device;
  if (body)
    new_stmt->body = body->clone();
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
