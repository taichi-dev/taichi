#include "taichi/program/async_utils.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/ir_bank.h"
#include "taichi/program/kernel.h"

TLANG_NAMESPACE_BEGIN

std::unique_ptr<IRNode> IRHandle::clone() const {
  TI_AUTO_PROF
  // TODO: remove get_kernel() here
  return irpass::analysis::clone(const_cast<IRNode *>(ir_), ir_->get_kernel());
}

TaskLaunchRecord::TaskLaunchRecord() : kernel(nullptr), ir_handle(nullptr, 0) {
}

std::atomic<int> TaskLaunchRecord::task_counter = 0;

TaskLaunchRecord::TaskLaunchRecord(Context context,
                                   Kernel *kernel,
                                   IRHandle ir_handle)
    : context(context), kernel(kernel), ir_handle(ir_handle) {
  id = task_counter++;
  TI_ASSERT(ir_handle.ir()->get_kernel() != nullptr);
}

OffloadedStmt *TaskLaunchRecord::stmt() const {
  TI_ASSERT(ir_handle.ir());
  return const_cast<IRNode *>(ir_handle.ir())->as<OffloadedStmt>();
}

bool TaskLaunchRecord::empty() const {
  return ir_handle.ir() == nullptr;
}

void TaskMeta::print() const {
  fmt::print("TaskMeta\n  name {}\n", name);
  fmt::print("  type {}\n", offloaded_task_type_name(type));
  if (snode != nullptr) {
    fmt::print("  snode {}\n", snode->get_node_type_name_hinted());
  } else {
    fmt::print("  snode nullptr\n");
  }
  if (!input_states.empty()) {
    fmt::print("  input states:\n    ");
    for (auto s : input_states) {
      fmt::print("{} ", s.name());
    }
    fmt::print("\n");
  }
  if (!output_states.empty()) {
    fmt::print("  output states:\n    ");
    for (auto s : output_states) {
      fmt::print("{} ", s.name());
    }
    fmt::print("\n");
  }
  std::vector<SNode *> element_wise_snodes, non_element_wise_snodes;
  for (auto s : element_wise) {
    if (s.second) {
      element_wise_snodes.push_back(s.first);
    } else {
      non_element_wise_snodes.push_back(s.first);
    }
  }
  if (!element_wise_snodes.empty()) {
    fmt::print("  element-wise snodes:\n    ");
    for (auto s : element_wise_snodes) {
      fmt::print("{} ", s->get_node_type_name_hinted());
    }
    fmt::print("\n");
  }
  if (!non_element_wise_snodes.empty()) {
    fmt::print("  non-element-wise snodes:\n    ");
    for (auto s : non_element_wise_snodes) {
      fmt::print("{} ", s->get_node_type_name_hinted());
    }
    fmt::print("\n");
  }
}

TaskMeta *get_task_meta(IRBank *ir_bank, const TaskLaunchRecord &t) {
  TI_AUTO_PROF
  // TODO: this function should ideally take only an IRNode
  static std::mutex mut;

  std::lock_guard<std::mutex> guard(mut);

  auto &meta_bank = ir_bank->meta_bank_;

  if (meta_bank.find(t.ir_handle) != meta_bank.end()) {
    return &meta_bank[t.ir_handle];
  }

  using namespace irpass::analysis;
  TaskMeta meta;
  // TODO: this is an abuse since it gathers nothing...
  auto *root_stmt = t.stmt();
  meta.name =
      t.kernel->name + "_" + offloaded_task_type_name(root_stmt->task_type);
  meta.type = root_stmt->task_type;
  get_meta_input_value_states(root_stmt, &meta);
  gather_statements(root_stmt, [&](Stmt *stmt) {
    if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (auto ptr = global_store->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.output_states.emplace(snode, AsyncState::Type::value);
        }
      }
    }
    if (auto global_atomic = stmt->cast<AtomicOpStmt>()) {
      if (auto ptr = global_atomic->dest->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.output_states.emplace(snode, AsyncState::Type::value);
        }
      }
    }

    if (auto *snode_op = stmt->cast<SNodeOpStmt>()) {
      if (snode_op->op_type == SNodeOpType::activate ||
          snode_op->op_type == SNodeOpType::deactivate) {
        auto *sn = snode_op->snode;
        if (is_gc_able(sn->type)) {
          meta.input_states.emplace(sn, AsyncState::Type::allocator);
          meta.input_states.emplace(sn, AsyncState::Type::mask);
          meta.output_states.emplace(sn, AsyncState::Type::allocator);
          meta.output_states.emplace(sn, AsyncState::Type::mask);
        }
      }
    }

    if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
      if (ptr->activate) {
        for (auto &snode : ptr->snodes.data) {
          auto s = snode;
          while (s) {
            if (!s->is_path_all_dense) {
              meta.input_states.emplace(s, AsyncState::Type::mask);
              meta.output_states.emplace(s, AsyncState::Type::mask);
              if (is_gc_able(s->type)) {
                meta.input_states.emplace(s, AsyncState::Type::allocator);
                meta.output_states.emplace(s, AsyncState::Type::allocator);
              }
            }
            s = s->parent;
          }
        }
      }
      for (auto &snode : ptr->snodes.data) {
        if (ptr->is_element_wise(snode)) {
          if (meta.element_wise.find(snode) == meta.element_wise.end()) {
            meta.element_wise[snode] = true;
          }
        } else {
          meta.element_wise[snode] = false;
        }
      }
    }
    if (auto clear_list = stmt->cast<ClearListStmt>()) {
      meta.output_states.emplace(clear_list->snode, AsyncState::Type::list);
    }
    // TODO: handle SNodeOpStmt etc.
    return false;
  });

  // We are being conservative here: if there are any non-element-wise
  // accesses (e.g., a = x[i + 1]), we don't treat it as completely
  // overwriting the value state (e.g., for i in x: x[i] = 0).
  for (auto &state : meta.output_states) {
    if (state.type == AsyncState::Type::value) {
      if (meta.element_wise.find(state.snode) == meta.element_wise.end()) {
        meta.input_states.insert(state);
      }
    }
  }

  if (root_stmt->task_type == OffloadedTaskType::listgen) {
    TI_ASSERT(root_stmt->snode->parent);
    meta.snode = root_stmt->snode;
    meta.input_states.emplace(root_stmt->snode->parent, AsyncState::Type::list);
    meta.input_states.emplace(root_stmt->snode, AsyncState::Type::list);
    meta.input_states.emplace(root_stmt->snode, AsyncState::Type::mask);
    meta.output_states.emplace(root_stmt->snode, AsyncState::Type::list);
  } else if (root_stmt->task_type == OffloadedTaskType::struct_for) {
    meta.snode = root_stmt->snode;
    meta.input_states.emplace(root_stmt->snode, AsyncState::Type::list);
  } else if ((root_stmt->task_type == OffloadedTaskType::gc) &&
             (is_gc_able(root_stmt->snode->type))) {
    meta.snode = root_stmt->snode;
    meta.input_states.emplace(meta.snode, AsyncState::Type::allocator);
    meta.output_states.emplace(meta.snode, AsyncState::Type::allocator);
  }

  meta_bank[t.ir_handle] = meta;
  return &meta_bank[t.ir_handle];
}

TaskFusionMeta get_task_fusion_meta(IRBank *bank, const TaskLaunchRecord &t) {
  TI_AUTO_PROF
  // TODO: this function should ideally take only an IRNode
  auto &fusion_meta_bank = bank->fusion_meta_bank_;
  if (fusion_meta_bank.find(t.ir_handle) != fusion_meta_bank.end()) {
    return fusion_meta_bank[t.ir_handle];
  }

  TaskFusionMeta meta{};
  if (t.kernel->is_accessor) {
    // SNode accessors can't be fused.
    // TODO: just avoid snode accessors going into the async engine
    return fusion_meta_bank[t.ir_handle] = TaskFusionMeta();
  }
  meta.kernel = t.kernel;
  if (t.kernel->args.empty() && t.kernel->rets.empty()) {
    meta.kernel = nullptr;
  }

  auto *task = t.stmt();
  meta.type = task->task_type;
  if (task->task_type == OffloadedTaskType::struct_for) {
    meta.snode = task->snode;
    meta.block_dim = task->block_dim;
  } else if (task->task_type == OffloadedTaskType::range_for) {
    // TODO: a few problems with the range-for test condition:
    // 1. This could incorrectly fuse two range-for kernels that have
    // different sizes, but then the loop ranges get padded to the same
    // power-of-two (E.g. maybe a side effect when a struct-for is demoted
    // to range-for).
    // 2. It has also fused range-fors that have the same linear range,
    // but are of different dimensions of loop indices, e.g. (16, ) and
    // (4, 4).
    if (!task->const_begin || !task->const_end) {
      // Do not fuse range-for tasks with variable ranges for now.
      return fusion_meta_bank[t.ir_handle] = TaskFusionMeta();
    }
    meta.begin_value = task->begin_value;
    meta.end_value = task->end_value;
  } else if (task->task_type != OffloadedTaskType::serial) {
    // Do not fuse gc/listgen tasks.
    meta.fusible = false;
    meta.snode = task->snode;
    return fusion_meta_bank[t.ir_handle] = meta;
  }
  meta.fusible = true;
  return fusion_meta_bank[t.ir_handle] = meta;
}

TLANG_NAMESPACE_END
