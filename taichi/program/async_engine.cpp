#include "taichi/program/async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/util/testing.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

uint64 hash(IRNode *stmt) {
  TI_ASSERT(stmt);
  // TODO: upgrade this using IR comparisons
  std::string serialized;
  irpass::re_id(stmt);
  irpass::print(stmt, &serialized);
  uint64 ret = 0;
  for (uint64 i = 0; i < serialized.size(); i++) {
    ret = ret * 100000007UL + (uint64)serialized[i];
  }
  return ret;
}

KernelLaunchRecord::KernelLaunchRecord(Context context,
                                       Kernel *kernel,
                                       std::unique_ptr<IRNode> &&stmt_)
    : context(context),
      kernel(kernel),
      stmt(dynamic_cast<OffloadedStmt *>(stmt_.get())),
      stmt_(std::move(stmt_)),
      h(hash(stmt)) {
  TI_ASSERT(stmt != nullptr);
}

void ExecutionQueue::enqueue(KernelLaunchRecord &&ker) {
  auto h = ker.h;
  auto stmt = ker.stmt;
  auto kernel = ker.kernel;
  if (compiled_func.find(h) == compiled_func.end() &&
      to_be_compiled.find(h) == to_be_compiled.end()) {
    to_be_compiled.insert(h);
    compilation_workers.enqueue([&, stmt, kernel, h, this]() {
      {
        // Final lowering
        using namespace irpass;

        flag_access(stmt);
        lower_access(stmt, true, kernel);
        flag_access(stmt);
        full_simplify(stmt, kernel->program.config, kernel);
        // analysis::verify(stmt);
      }
      auto func = CodeGenCPU(kernel, stmt).codegen();
      std::lock_guard<std::mutex> _(mut);
      compiled_func[h] = func;
    });
  }

  auto context = ker.context;
  launch_worker.enqueue([&, h, stmt, context, this] {
    FunctionType func;
    while (true) {
      std::unique_lock<std::mutex> lock(mut);
      if (compiled_func.find(h) == compiled_func.end()) {
        lock.unlock();
        Time::sleep(1e-6);
        continue;
      }
      func = compiled_func[h];
      break;
    }
    stat.add("launched_kernels", 1.0);
    auto task_type = stmt->task_type;
    if (task_type == OffloadedStmt::TaskType::listgen) {
      stat.add("launched_kernels_list_op", 1.0);
      stat.add("launched_kernels_list_gen", 1.0);
    } else if (task_type == OffloadedStmt::TaskType::clear_list) {
      stat.add("launched_kernels_list_op", 1.0);
      stat.add("launched_kernels_list_clear", 1.0);
    } else if (task_type == OffloadedStmt::TaskType::range_for) {
      stat.add("launched_kernels_compute", 1.0);
      stat.add("launched_kernels_range_for", 1.0);
    } else if (task_type == OffloadedStmt::TaskType::struct_for) {
      stat.add("launched_kernels_compute", 1.0);
      stat.add("launched_kernels_struct_for", 1.0);
    } else if (task_type == OffloadedStmt::TaskType::gc) {
      stat.add("launched_kernels_garbage_collect", 1.0);
    }
    auto c = context;
    func(c);
  });
  trashbin.push_back(std::move(ker));
}

void ExecutionQueue::synchronize() {
  TI_AUTO_PROF
  launch_worker.flush();
}

ExecutionQueue::ExecutionQueue()
    : compilation_workers(4), launch_worker(1) {  // TODO: remove 4
}

void AsyncEngine::launch(Kernel *kernel) {
  if (!kernel->lowered)
    kernel->lower(false);
  auto block = dynamic_cast<Block *>(kernel->ir);
  TI_ASSERT(block);
  auto &offloads = block->statements;
  for (std::size_t i = 0; i < offloads.size(); i++) {
    auto offload = offloads[i]->as<OffloadedStmt>();
    KernelLaunchRecord rec(kernel->program.get_context(), kernel,
                           irpass::analysis::clone(offload, kernel));
    enqueue(std::move(rec));
  }
}

void AsyncEngine::enqueue(KernelLaunchRecord &&t) {
  using namespace irpass::analysis;

  auto &meta = metas[t.h];
  // TODO: this is an abuse since it gathers nothing...
  auto root_stmt = t.stmt;
  gather_statements(root_stmt, [&](Stmt *stmt) {
    if (auto global_ptr = stmt->cast<GlobalPtrStmt>()) {
      for (auto &snode : global_ptr->snodes.data) {
        meta.input_snodes.insert(snode);
      }
    }
    if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      if (auto ptr = global_load->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_snodes.insert(snode);
        }
      }
    }
    if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (auto ptr = global_store->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.output_snodes.insert(snode);
        }
      }
    }
    if (auto global_atomic = stmt->cast<AtomicOpStmt>()) {
      if (auto ptr = global_atomic->dest->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_snodes.insert(snode);
          meta.output_snodes.insert(snode);
        }
      }
    }

    if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
      if (ptr->activate) {
        for (auto &snode : ptr->snodes.data) {
          meta.activation_snodes.insert(snode);
          // fmt::print(" **** act {}\n", snode->get_node_type_name_hinted());
        }
      }
    }
    return false;
  });

  task_queue.push_back(std::move(t));
}

void AsyncEngine::synchronize() {
  optimize_listgen();
  while (fuse())
    ;
  while (!task_queue.empty()) {
    queue.enqueue(std::move(task_queue.front()));
    task_queue.pop_front();
  }
  queue.synchronize();
}

bool AsyncEngine::optimize_listgen() {
  // TODO: improve...
  bool modified = false;
  std::unordered_map<SNode *, bool> list_dirty;
  auto new_task_queue = std::deque<KernelLaunchRecord>();
  for (int i = 0; i < task_queue.size(); i++) {
    // Try to eliminate unused listgens
    auto &t = task_queue[i];
    auto meta = metas[t.h];
    auto offload = t.stmt;
    bool keep = true;
    if (offload->task_type == OffloadedStmt::TaskType::listgen) {
      // keep
    } else if (offload->task_type == OffloadedStmt::TaskType::clear_list) {
      TI_ASSERT(task_queue[i + 1].stmt->task_type ==
                OffloadedStmt::TaskType::listgen);
      auto snode = offload->snode;
      if (list_dirty.find(snode) != list_dirty.end() && !list_dirty[snode]) {
        keep = false;  // safe to remove
        modified = true;
        i++;  // skip the following list gen as well
        continue;
      }
      list_dirty[snode] = false;
    } else {
      for (auto snode : meta.activation_snodes) {
        while (snode && snode->type != SNodeType::root) {
          list_dirty[snode] = true;
          snode = snode->parent;
        }
      }
    }
    if (keep) {
      new_task_queue.push_back(std::move(t));
    } else {
      modified = true;
    }
  }
  task_queue = std::move(new_task_queue);
  return modified;
}

bool AsyncEngine::fuse() {
  // TODO: improve...
  bool modified = false;
  std::unordered_map<SNode *, bool> list_dirty;

  if (false) {
    // (experimental) print tasks
    for (int i = 0; i < (int)task_queue.size(); i++) {
      fmt::print("{}: {}\n", i, task_queue[i].stmt->task_name());
      irpass::print(task_queue[i].stmt);
    }
  }

  for (int i = 0; i < (int)task_queue.size() - 1; i++) {
    auto task_a = task_queue[i].stmt;
    auto task_b = task_queue[i + 1].stmt;
    bool is_same_struct_for = task_a->task_type == OffloadedStmt::struct_for &&
                              task_b->task_type == OffloadedStmt::struct_for &&
                              task_a->snode == task_b->snode &&
                              task_a->block_dim == task_b->block_dim;
    bool is_same_range_for = task_a->task_type == OffloadedStmt::range_for &&
                             task_b->task_type == OffloadedStmt::range_for &&
                             task_a->const_begin && task_b->const_begin &&
                             task_a->const_end && task_b->const_end &&
                             task_a->begin_value == task_b->begin_value &&
                             task_a->end_value == task_b->end_value;

    // We do not fuse serial kernels for now since they can be SNode accessors
    bool are_both_serial = task_a->task_type == OffloadedStmt::serial &&
                           task_b->task_type == OffloadedStmt::serial;
    bool same_kernel = task_queue[i].kernel == task_queue[i + 1].kernel;
    if (is_same_range_for || is_same_struct_for) {
      // TODO: in certain cases this optimization can be wrong!
      // Fuse task b into task_a
      for (int j = 0; j < (int)task_b->body->size(); j++) {
        task_a->body->insert(std::move(task_b->body->statements[j]));
      }
      task_b->body->statements.clear();

      // replace all reference to the offloaded statement B to A
      irpass::replace_all_usages_with(task_a, task_b, task_a);
      irpass::re_id(task_a);
      irpass::fix_block_parents(task_a);

      auto kernel = task_queue[i].kernel;
      irpass::full_simplify(task_a, kernel->program.config, kernel);
      task_queue[i].h = hash(task_a);

      modified = true;
    }
  }

  auto new_task_queue = std::deque<KernelLaunchRecord>();

  // Eliminate empty tasks
  for (int i = 0; i < (int)task_queue.size(); i++) {
    auto task = task_queue[i].stmt;
    bool keep = true;
    if (task->task_type == OffloadedStmt::struct_for ||
        task->task_type == OffloadedStmt::range_for ||
        task->task_type == OffloadedStmt::serial) {
      if (task->body->statements.empty())
        keep = false;
    }
    if (keep) {
      new_task_queue.push_back(std::move(task_queue[i]));
    }
  }

  task_queue = std::move(new_task_queue);

  return modified;
}

TLANG_NAMESPACE_END
