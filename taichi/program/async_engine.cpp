#include "taichi/program/async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/util/testing.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/program/extension.h"

TLANG_NAMESPACE_BEGIN

namespace {

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

std::unique_ptr<OffloadedStmt> clone_offloaded_task(OffloadedStmt *from,
                                                    Kernel *kernel) {
  auto new_ir = irpass::analysis::clone(from, kernel);
  return std::unique_ptr<OffloadedStmt>((OffloadedStmt *)(new_ir.release()));
}

}  // namespace

ParallelExecutor::ParallelExecutor(int num_threads)
    : num_threads(num_threads),
      status(ExecutorStatus::uninitialized),
      running_threads(0) {
  {
    auto _ = std::lock_guard<std::mutex>(mut);

    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back([this]() { this->worker_loop(); });
    }

    status = ExecutorStatus::initialized;
  }
  init_cv_.notify_all();
}

ParallelExecutor::~ParallelExecutor() {
  // TODO: We should have a new ExecutorStatus, e.g. shutting_down, to prevent
  // new tasks from being enqueued during shut down.
  flush();
  {
    auto _ = std::lock_guard<std::mutex>(mut);
    status = ExecutorStatus::finalized;
  }
  // Signal the workers that they need to shutdown.
  worker_cv_.notify_all();
  for (auto &th : threads) {
    th.join();
  }
}

void ParallelExecutor::enqueue(const TaskType &func) {
  {
    std::lock_guard<std::mutex> _(mut);
    task_queue.push_back(func);
  }
  worker_cv_.notify_all();
}

void ParallelExecutor::flush() {
  std::unique_lock<std::mutex> lock(mut);
  while (!flush_cv_cond()) {
    flush_cv_.wait(lock);
  }
}

bool ParallelExecutor::flush_cv_cond() {
  return (task_queue.empty() && running_threads == 0);
}

void ParallelExecutor::worker_loop() {
  TI_DEBUG("Starting worker thread.");
  {
    std::unique_lock<std::mutex> lock(mut);
    while (status == ExecutorStatus::uninitialized) {
      init_cv_.wait(lock);
    }
  }

  TI_DEBUG("Worker thread initialized and running.");
  bool done = false;
  while (!done) {
    bool notify_flush_cv = false;
    {
      std::unique_lock<std::mutex> lock(mut);
      while (task_queue.empty() && status == ExecutorStatus::initialized) {
        worker_cv_.wait(lock);
      }
      // So long as |task_queue| is not empty, we keep running.
      if (!task_queue.empty()) {
        auto task = task_queue.front();
        running_threads++;
        task_queue.pop_front();
        lock.unlock();

        // Run the task
        task();

        lock.lock();
        running_threads--;
      }
      notify_flush_cv = flush_cv_cond();
      if (status == ExecutorStatus::finalized && task_queue.empty()) {
        done = true;
      }
    }
    if (notify_flush_cv) {
      // It is fine to notify |flush_cv_| while nobody is waiting on it.
      flush_cv_.notify_one();
    }
  }
}

TaskLaunchRecord::TaskLaunchRecord(Context context,
                                   Kernel *kernel,
                                   OffloadedStmt *stmt,
                                   uint64 h)
    : context(context),
      kernel(kernel),
      h(h),
      stmt_(stmt),
      cloned_stmt_holder_(nullptr) {
  TI_ASSERT(stmt_ != nullptr);
  TI_ASSERT(stmt_->get_kernel() != nullptr);
}

OffloadedStmt *TaskLaunchRecord::clone_stmt_on_write() {
  if (cloned_stmt_holder_ == nullptr) {
    cloned_stmt_holder_ = clone_offloaded_task(stmt_, kernel);
    stmt_ = cloned_stmt_holder_.get();
  }
  return stmt_;
}

void ExecutionQueue::enqueue(TaskLaunchRecord &&ker) {
  auto h = ker.h;
  auto *stmt = ker.stmt();
  auto kernel = ker.kernel;

  bool needs_compile = false;
  AsyncCompiledFunc *async_func = nullptr;
  {
    std::lock_guard<std::mutex> _(mut);
    needs_compile = (compiled_funcs_.find(h) == compiled_funcs_.end());
    if (needs_compile) {
      compiled_funcs_.emplace(h, AsyncCompiledFunc());
    }
    async_func = &(compiled_funcs_.at(h));
  }
  if (needs_compile) {
    // Later the IR passes will change |stmt|, so we must clone it.
    stmt = ker.clone_stmt_on_write();

    compilation_workers.enqueue([async_func, stmt, kernel]() {
      {
        // Final lowering
        using namespace irpass;

        auto config = kernel->program.config;
        auto ir = stmt;
        offload_to_executable(
            ir, config, /*verbose=*/false,
            /*lower_global_access=*/true,
            /*make_thread_local=*/true,
            /*make_block_local=*/
            is_extension_supported(config.arch, Extension::bls) &&
                config.make_block_local);
      }
      auto codegen = KernelCodeGen::create(kernel->arch, kernel, stmt);
      auto func = codegen->codegen();
      async_func->set(func);
    });
  }

  kernel->account_for_offloaded(ker.stmt());

  launch_worker.enqueue([async_func, context = ker.context]() mutable {
    auto func = async_func->get();
    func(context);
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

void AsyncEngine::launch(Kernel *kernel, Context &context) {
  if (!kernel->lowered)
    kernel->lower(/*to_executable=*/false);

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  TI_ASSERT(block);

  auto &offloads = block->statements;
  auto &kmeta = kernel_metas_[kernel];
  const bool kmeta_inited = kmeta.initialized;
  for (std::size_t i = 0; i < offloads.size(); i++) {
    auto *offload = offloads[i]->as<OffloadedStmt>();
    uint64 h;
    OffloadedStmt *offl_template = nullptr;
    if (kmeta_inited) {
      auto &oc = kmeta.offloaded_cached[i];
      h = oc.get_hash();
      offl_template = oc.get_template();
    } else {
      auto cloned_offs = clone_offloaded_task(offload, kernel);
      offl_template = cloned_offs.get();
      h = hash(offl_template);
      TI_ASSERT(kmeta.offloaded_cached.size() == i);
      kmeta.offloaded_cached.emplace_back(std::move(cloned_offs), h);
    }
    TaskLaunchRecord rec(context, kernel, offl_template, h);
    enqueue(std::move(rec));
  }
  if (!kmeta_inited) {
    kmeta.initialized = true;
  }
}

TaskMeta AsyncEngine::create_task_meta(
    const taichi::lang::TaskLaunchRecord &t) {
  using namespace irpass::analysis;
  TaskMeta meta;
  // TODO: this is an abuse since it gathers nothing...
  auto *root_stmt = t.stmt();
  meta.kernel_name = t.kernel->name + "_" +
                     OffloadedStmt::task_type_name(root_stmt->task_type);
  gather_statements(root_stmt, [&](Stmt *stmt) {
    if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      if (auto ptr = global_load->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_states.emplace_back(snode, AsyncState::Type::value);
        }
      }
    }
    if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (auto ptr = global_store->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.output_states.emplace_back(snode, AsyncState::Type::value);
          if (ptr->activate)
            meta.output_states.emplace_back(snode, AsyncState::Type::mask);
        }
      }
    }
    if (auto global_atomic = stmt->cast<AtomicOpStmt>()) {
      if (auto ptr = global_atomic->dest->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_states.emplace_back(snode, AsyncState::Type::value);
          meta.output_states.emplace_back(snode, AsyncState::Type::value);
          if (ptr->activate)
            meta.output_states.emplace_back(snode, AsyncState::Type::mask);
        }
      }
    }

    if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
      if (ptr->activate) {
        for (auto &snode : ptr->snodes.data) {
          meta.output_states.emplace_back(snode, AsyncState::Type::mask);
        }
      }
    }
    return false;
  });
  if (root_stmt->task_type == OffloadedStmt::listgen) {
    meta.input_states.emplace_back(root_stmt->snode, AsyncState::Type::list);
    meta.input_states.emplace_back(root_stmt->snode, AsyncState::Type::mask);
    meta.output_states.emplace_back(root_stmt->snode, AsyncState::Type::list);
  } else if (root_stmt->task_type == OffloadedStmt::struct_for) {
    meta.input_states.emplace_back(root_stmt->snode, AsyncState::Type::list);
  } else if (root_stmt->task_type == OffloadedStmt::clear_list) {
    meta.input_states.emplace_back(root_stmt->snode, AsyncState::Type::list);
    meta.output_states.emplace_back(root_stmt->snode, AsyncState::Type::list);
  }
  // TODO: this is probably not fully done. Hopefully after SFG Graphviz is
  // done we can easily spot what's left.
  return meta;
}

void AsyncEngine::enqueue(TaskLaunchRecord &&t) {
  if (offloaded_metas_.find(t.h) == offloaded_metas_.end()) {
    offloaded_metas_[t.h] = create_task_meta(t);
  }
  sfg->insert_task(offloaded_metas_[t.h]);
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
  auto new_task_queue = std::deque<TaskLaunchRecord>();
  for (int i = 0; i < task_queue.size(); i++) {
    // Try to eliminate unused listgens
    auto &t = task_queue[i];
    auto meta = offloaded_metas_[t.h];
    const auto *offload = t.stmt();
    bool keep = true;
    if (offload->task_type == OffloadedStmt::TaskType::listgen) {
      // keep
    } else if (offload->task_type == OffloadedStmt::TaskType::clear_list) {
      TI_ASSERT(task_queue[i + 1].stmt()->task_type ==
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
      for (auto output_state : meta.output_states) {
        auto snode = output_state.snode;
        if (output_state.type != AsyncState::Type::mask)
          continue;
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
      fmt::print("{}: {}\n", i, task_queue[i].stmt()->task_name());
      irpass::print(task_queue[i].stmt());
    }
  }

  for (int i = 0; i < (int)task_queue.size() - 1; i++) {
    auto &rec_a = task_queue[i];
    auto &rec_b = task_queue[i + 1];
    auto *task_a = rec_a.stmt();
    auto *task_b = rec_b.stmt();
    bool is_same_struct_for = task_a->task_type == OffloadedStmt::struct_for &&
                              task_b->task_type == OffloadedStmt::struct_for &&
                              task_a->snode == task_b->snode &&
                              task_a->block_dim == task_b->block_dim;
    // TODO: a few problems with the range-for test condition:
    // 1. This could incorrectly fuse two range-for kernels that have different
    // sizes, but then the loop ranges get padded to the same power-of-two (E.g.
    // maybe a side effect when a struct-for is demoted to range-for).
    // 2. It has also fused range-fors that have the same linear range, but are
    // of different dimensions of loop indices, e.g. (16, ) and (4, 4).
    bool is_same_range_for = task_a->task_type == OffloadedStmt::range_for &&
                             task_b->task_type == OffloadedStmt::range_for &&
                             task_a->const_begin && task_b->const_begin &&
                             task_a->const_end && task_b->const_end &&
                             task_a->begin_value == task_b->begin_value &&
                             task_a->end_value == task_b->end_value;

    // We do not fuse serial kernels for now since they can be SNode accessors
    bool are_both_serial = task_a->task_type == OffloadedStmt::serial &&
                           task_b->task_type == OffloadedStmt::serial;
    const bool same_kernel = (rec_a.kernel == rec_b.kernel);
    bool kernel_args_match = true;
    if (!same_kernel) {
      // Merging kernels with different signatures will break invariants. E.g.
      // https://github.com/taichi-dev/taichi/blob/a6575fb97557267e2f550591f43b183076b72ac2/taichi/transforms/type_check.cpp#L326
      //
      // TODO: we could merge different kernels if their args are the same. But
      // we have no way to check that for now.
      auto check = [](const Kernel *k) {
        return (k->args.empty() && k->rets.empty());
      };
      kernel_args_match = (check(rec_a.kernel) && check(rec_b.kernel));
    }
    if (kernel_args_match && (is_same_range_for || is_same_struct_for)) {
      // We are about to change both |task_a| and |task_b|. Clone them first.
      task_a = rec_a.clone_stmt_on_write();
      task_b = rec_b.clone_stmt_on_write();
      // TODO: in certain cases this optimization can be wrong!
      // Fuse task b into task_a
      for (int j = 0; j < (int)task_b->body->size(); j++) {
        task_a->body->insert(std::move(task_b->body->statements[j]));
      }
      task_b->body->statements.clear();

      // replace all reference to the offloaded statement B to A
      irpass::replace_all_usages_with(task_a, task_b, task_a);
      irpass::re_id(task_a);

      auto kernel = task_queue[i].kernel;
      irpass::full_simplify(task_a, /*after_lower_access=*/false, kernel);
      task_queue[i].h = hash(task_a);

      modified = true;
    }
  }

  auto new_task_queue = std::deque<TaskLaunchRecord>();

  // Eliminate empty tasks
  for (int i = 0; i < (int)task_queue.size(); i++) {
    auto *task = task_queue[i].stmt();
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
