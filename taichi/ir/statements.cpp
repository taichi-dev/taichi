// TODO: gradually cppize statements.h
#include "statements.h"
#include "taichi/program/program.h"

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
  block_dim = 0;
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

TLANG_NAMESPACE_END
