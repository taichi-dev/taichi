#include "taichi/program/async_utils.h"

#include <queue>
#include <unordered_map>

#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/ir_bank.h"
#include "taichi/program/kernel.h"

// Keep this include in the end!
#include "taichi/program/async_profiler_switch.h"

TLANG_NAMESPACE_BEGIN

std::unique_ptr<IRNode> IRHandle::clone() const {
  TI_AUTO_PROF
  // TODO: remove get_kernel() here
  return irpass::analysis::clone(const_cast<IRNode *>(ir_), ir_->get_kernel());
}

TaskLaunchRecord::TaskLaunchRecord() : kernel(nullptr), ir_handle(nullptr, 0) {
}

// Initial node has rec.id == 0, so we start from rec.id == 1.
std::atomic<int> TaskLaunchRecord::task_counter = 1;

TaskLaunchRecord::TaskLaunchRecord(RuntimeContext context,
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

std::string AsyncState::name() const {
  std::string type_name;
  switch (type) {
    case Type::mask:
      type_name = "mask";
      break;
    case Type::value:
      type_name = "value";
      break;
    case Type::list:
      type_name = "list";
      break;
    case Type::allocator:
      type_name = "allocator";
      break;
    case Type::undefined:
      TI_ERROR("invalue type");
  }
  const auto prefix =
      holds_snode()
          ? std::get<SNode *>(snode_or_global_tmp)->get_node_type_name_hinted()
          : fmt::format("global_tmp[{}]",
                        std::get<Kernel *>(snode_or_global_tmp)->name);
  return prefix + "_" + type_name;
}

std::size_t AsyncState::perfect_hash(void *ptr, AsyncState::Type type) {
  static_assert((int)Type::undefined < 8);
  static_assert(std::alignment_of<SNode>() % 8 == 0);
  static_assert(std::alignment_of<Kernel>() % 8 == 0);
  return (std::size_t)ptr ^ (std::size_t)type;
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
  if (!loop_unique.empty()) {
    fmt::print("  loop-unique snodes:\n    ");
    for (auto &s : loop_unique) {
      fmt::print("{}:{} ", s.first->get_node_type_name_hinted(),
                 s.second ? s.second->name() : "nullptr");
    }
    fmt::print("\n");
  }
  std::vector<const SNode *> element_wise_snodes, non_element_wise_snodes;
  for (auto &s : element_wise) {
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
  // TODO: this function should ideally take only an IRNode
  static std::mutex mut;

  std::lock_guard<std::mutex> guard(mut);

  auto &meta_bank = ir_bank->meta_bank_;

  if (meta_bank.find(t.ir_handle) != meta_bank.end()) {
    return &meta_bank[t.ir_handle];
  }

  using namespace irpass::analysis;
  TaskMeta meta;
  auto *root_stmt = t.stmt();
  meta.name =
      t.kernel->name + "_" + offloaded_task_type_name(root_stmt->task_type);
  meta.type = root_stmt->task_type;
  get_meta_input_value_states(root_stmt, &meta, ir_bank);
  meta.loop_unique = gather_uniquely_accessed_pointers(root_stmt);

  std::unordered_set<SNode *> activates, deactivates;

  // TODO: this is an abuse since it gathers nothing...
  gather_statements(root_stmt, [&](Stmt *stmt) {
    // For a global load, GlobalPtrStmt has already been handled in
    // get_meta_input_value_states().
    if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (auto dest = global_store->dest->cast<GlobalPtrStmt>()) {
        for (auto &snode : dest->snodes.data) {
          meta.output_states.insert(
              ir_bank->get_async_state(snode, AsyncState::Type::value));
        }
      }
      if (auto global_tensor_element =
              global_store->dest->cast<PtrOffsetStmt>()) {
        if (global_tensor_element->is_unlowered_global_ptr()) {
          if (auto dest =
                  global_tensor_element->origin->cast<GlobalPtrStmt>()) {
            for (auto &snode : dest->snodes.data) {
              meta.output_states.insert(
                  ir_bank->get_async_state(snode, AsyncState::Type::value));
            }
          }
        }
      }
    }
    if (auto global_atomic = stmt->cast<AtomicOpStmt>()) {
      if (auto dest = global_atomic->dest->cast<GlobalPtrStmt>()) {
        for (auto &snode : dest->snodes.data) {
          // input_state is already handled in
          // get_meta_input_value_states().
          meta.output_states.insert(
              ir_bank->get_async_state(snode, AsyncState::Type::value));
        }
      }
    }

    if (auto *snode_op = stmt->cast<SNodeOpStmt>()) {
      auto *sn = snode_op->snode;
      const auto sty = snode_op->op_type;
      if (sty == SNodeOpType::activate) {
        activates.insert(sn);
      } else if (sty == SNodeOpType::deactivate) {
        deactivates.insert(sn);
      } else if (snode_op->op_type == SNodeOpType::append) {
        activates.insert(sn);
        for (auto &child : sn->ch) {
          TI_ASSERT(child->type == SNodeType::place);
          meta.input_states.insert(
              ir_bank->get_async_state(child.get(), AsyncState::Type::value));
          meta.output_states.insert(
              ir_bank->get_async_state(child.get(), AsyncState::Type::value));
        }
      } else if (snode_op->op_type == SNodeOpType::is_active ||
                 snode_op->op_type == SNodeOpType::length) {
        meta.input_states.insert(
            ir_bank->get_async_state(sn, AsyncState::Type::mask));
      } else if (snode_op->op_type == SNodeOpType::get_addr) {
        // do nothing
      } else {
        TI_NOT_IMPLEMENTED
      }
    }

    if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
      if (ptr->activate) {
        for (auto &snode : ptr->snodes.data) {
          activates.insert(snode);
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
    if (stmt->is<GlobalTemporaryStmt>()) {
      auto as = ir_bank->get_async_state(t.kernel);
      meta.input_states.insert(as);
      meta.output_states.insert(as);
    }
    if (auto clear_list = stmt->cast<ClearListStmt>()) {
      meta.output_states.insert(
          ir_bank->get_async_state(clear_list->snode, AsyncState::Type::list));
    }
    return false;
  });

  std::unordered_set<SNode *> kernel_forces_no_activate(
      t.kernel->no_activate.begin(), t.kernel->no_activate.end());

  std::unordered_set<SNode *> mask_state_inserted;
  auto insert_mask_states_bottom_up = [&](SNode *s) {
    while (s) {
      if (kernel_forces_no_activate.count(s) > 0) {
        break;
      }
      if (mask_state_inserted.count(s) > 0) {
        // already handled by other activations
        break;
      }
      mask_state_inserted.insert(s);

      // Do not record dense SNodes' mask states.
      if (s->need_activation()) {
        meta.input_states.insert(
            ir_bank->get_async_state(s, AsyncState::Type::mask));
        meta.output_states.insert(
            ir_bank->get_async_state(s, AsyncState::Type::mask));
        if (is_gc_able(s->type)) {
          meta.input_states.insert(
              ir_bank->get_async_state(s, AsyncState::Type::allocator));
          meta.output_states.insert(
              ir_bank->get_async_state(s, AsyncState::Type::allocator));
        }
      }
      s = s->parent;
    }
  };

  for (auto &snode : activates) {
    insert_mask_states_bottom_up(snode);
  }
  for (auto &snode : deactivates) {
    insert_mask_states_bottom_up(snode);
  }

  auto insert_value_states_top_down = [&](SNode *snode) {
    // Insert output value states for all descendents of snode.
    // Input value states will be inserted later if it's not
    // element-wise written.
    std::queue<SNode *> to_insert;
    to_insert.push(snode);
    while (!to_insert.empty()) {
      auto *s = to_insert.front();
      to_insert.pop();
      if (kernel_forces_no_activate.count(s) > 0) {
        continue;
      }
      if (s->type == SNodeType::place) {
        meta.output_states.insert(
            ir_bank->get_async_state(s, AsyncState::Type::value));
      } else {
        for (auto &child : s->ch) {
          if (deactivates.count(child.get()) == 0) {
            // not already handled by other deactivations
            to_insert.push(child.get());
          }
        }
      }
    }
  };

  for (auto &snode : deactivates) {
    // The value states are actually modified in the next gc task of snode.
    insert_value_states_top_down(snode);
  }

  if (root_stmt->task_type == OffloadedTaskType::listgen) {
    TI_ASSERT(root_stmt->snode->parent);
    meta.snode = root_stmt->snode;
    meta.input_states.insert(ir_bank->get_async_state(root_stmt->snode->parent,
                                                      AsyncState::Type::list));
    meta.input_states.insert(
        ir_bank->get_async_state(root_stmt->snode, AsyncState::Type::list));
    if (root_stmt->snode->need_activation()) {
      meta.input_states.insert(
          ir_bank->get_async_state(root_stmt->snode, AsyncState::Type::mask));
    }
    meta.output_states.insert(
        ir_bank->get_async_state(root_stmt->snode, AsyncState::Type::list));
  } else if (root_stmt->task_type == OffloadedTaskType::struct_for) {
    meta.snode = root_stmt->snode;
    meta.input_states.insert(
        ir_bank->get_async_state(root_stmt->snode, AsyncState::Type::list));
  } else if ((root_stmt->task_type == OffloadedTaskType::gc) &&
             (is_gc_able(root_stmt->snode->type))) {
    meta.snode = root_stmt->snode;
    meta.input_states.insert(
        ir_bank->get_async_state(root_stmt->snode, AsyncState::Type::mask));
    meta.input_states.insert(ir_bank->get_async_state(
        root_stmt->snode, AsyncState::Type::allocator));
    meta.output_states.insert(
        ir_bank->get_async_state(root_stmt->snode, AsyncState::Type::mask));
    meta.output_states.insert(ir_bank->get_async_state(
        root_stmt->snode, AsyncState::Type::allocator));
    insert_value_states_top_down(root_stmt->snode);
  }

  for (auto &state : meta.output_states) {
    // We need to insert input value states in case of partial writes.
    // Assume we write sn on every indices we access in this task,
    // because we would have inserted the input value state in
    // get_meta_input_value_states otherwise.
    if (state.type == AsyncState::Type::value && state.holds_snode()) {
      const auto *sn = state.snode();
      bool completely_overwriting = false;
      if (meta.element_wise[sn]) {
        // If every access on sn is element-wise, then it must be
        // completely overwriting.
        completely_overwriting = true;
        // TODO: this is also completely overwriting although element_wise[sn]
        //  is false:
        // for i in x:
        //     x[i] = 0
        //     x[i + 1] = 0
        // A solution to this is to gather all definite writes in the task,
        // and check if any one of them ->covers_snode(sn).
        // TODO: is element-wise useless since it must be loop-unique?
      }
      if (meta.loop_unique.count(sn) > 0 && meta.loop_unique[sn] != nullptr) {
        if (meta.loop_unique[sn]->covers_snode(sn)) {
          completely_overwriting = true;
        }
      }
      if (!completely_overwriting) {
        meta.input_states.insert(state);
      }
    }
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
    // We don't need to record index_offsets because it's not used anymore.
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
