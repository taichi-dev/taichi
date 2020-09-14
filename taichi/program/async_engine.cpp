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
  // TODO: separate kernel from IR template
  serialized += stmt->get_kernel()->name;
  uint64 ret = 0;
  for (uint64 i = 0; i < serialized.size(); i++) {
    ret = ret * 100000007UL + (uint64)serialized[i];
  }
  return ret;
}

inline const SNode *get_snode_in_clear_list_task(const OffloadedStmt *task) {
  TI_ASSERT(is_clear_list_task(task));
  return task->body->back()->as<ClearListStmt>()->snode;
}

inline SNode *get_snode_in_clear_list_task(OffloadedStmt *task) {
  // Avoid duplication: https://stackoverflow.com/a/123995/12003165
  const auto *sn =
      get_snode_in_clear_list_task(static_cast<const OffloadedStmt *>(task));
  return const_cast<SNode *>(sn);
}

}  // namespace

uint64 IRBank::get_hash(IRNode *ir) {
  auto result_iterator = hash_bank_.find(ir);
  if (result_iterator == hash_bank_.end()) {
    auto result = hash(ir);
    set_hash(ir, result);
    return result;
  }
  return result_iterator->second;
}

void IRBank::set_hash(IRNode *ir, uint64 hash) {
  hash_bank_[ir] = hash;
}

bool IRBank::insert(std::unique_ptr<IRNode> &&ir, uint64 hash) {
  IRHandle handle(ir.get(), hash);
  auto insert_place = ir_bank_.find(handle);
  if (insert_place == ir_bank_.end()) {
    ir_bank_.emplace(handle, std::move(ir));
    return true;
  }
  insert_to_trash_bin(std::move(ir));
  return false;
}

void IRBank::insert_to_trash_bin(std::unique_ptr<IRNode> &&ir) {
  trash_bin.push_back(std::move(ir));
}

IRNode *IRBank::find(IRHandle ir_handle) {
  auto result = ir_bank_.find(ir_handle);
  if (result == ir_bank_.end())
    return nullptr;
  return result->second.get();
}

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

void ExecutionQueue::enqueue(const TaskLaunchRecord &ker) {
  auto h = ker.ir_handle.hash();
  auto *stmt = ker.stmt();
  auto kernel = ker.kernel;

  kernel->account_for_offloaded(stmt);

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
    auto cloned_stmt = ker.ir_handle.clone();
    stmt = cloned_stmt->as<OffloadedStmt>();

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
    ir_bank_->insert_to_trash_bin(std::move(cloned_stmt));
  }

  launch_worker.enqueue([async_func, context = ker.context]() mutable {
    auto func = async_func->get();
    func(context);
  });
}

void ExecutionQueue::synchronize() {
  TI_AUTO_PROF
  launch_worker.flush();
}

ExecutionQueue::ExecutionQueue(IRBank *ir_bank)
    : compilation_workers(4),  // TODO: remove 4
      launch_worker(1),
      ir_bank_(ir_bank) {
}

AsyncEngine::AsyncEngine(Program *program)
    : queue(&ir_bank_),
      program(program),
      sfg(std::make_unique<StateFlowGraph>()) {
}

void AsyncEngine::launch(Kernel *kernel, Context &context) {
  if (!kernel->lowered)
    kernel->lower(/*to_executable=*/false);

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  TI_ASSERT(block);

  auto &offloads = block->statements;
  auto &kmeta = kernel_metas_[kernel];
  const bool kmeta_inited = !kmeta.ir_handle_cached.empty();
  for (std::size_t i = 0; i < offloads.size(); i++) {
    if (!kmeta_inited) {
      TI_ASSERT(kmeta.ir_handle_cached.size() == i);
      IRHandle tmp_ir_handle(offloads[i].get(), 0);
      auto cloned_offs = tmp_ir_handle.clone();
      irpass::re_id(cloned_offs.get());
      auto h = ir_bank_.get_hash(cloned_offs.get());
      kmeta.ir_handle_cached.emplace_back(cloned_offs.get(), h);
      ir_bank_.insert(std::move(cloned_offs), h);
    }
    TaskLaunchRecord rec(context, kernel, kmeta.ir_handle_cached[i]);
    enqueue(rec);
  }
}

TaskMeta AsyncEngine::create_task_meta(const TaskLaunchRecord &t) {
  using namespace irpass::analysis;
  TaskMeta meta;
  // TODO: this is an abuse since it gathers nothing...
  auto *root_stmt = t.stmt();
  meta.kernel_name = t.kernel->name + "_" +
                     OffloadedStmt::task_type_name(root_stmt->task_type);
  meta.type = root_stmt->task_type;
  gather_statements(root_stmt, [&](Stmt *stmt) {
    if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      if (auto ptr = global_load->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_states.emplace(snode, AsyncState::Type::value);
        }
      }
    }

    // Note: since global store may only partially modify a value state, the
    // result (which contains the modified and unmodified part) actually needs a
    // read from the previous version of the value state.
    //
    // I.e.,
    // output_value_state = merge(input_value_state, written_part)
    //
    // Therefore we include the value state in input_states.
    //
    // The only exception is that the task may completely overwrite the value
    // state (e.g., for i in x: x[i] = 0). However, for now we are not yet
    // able to detect that case, so we are being conservative here.

    if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (auto ptr = global_store->ptr->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_states.emplace(snode, AsyncState::Type::value);
          meta.output_states.emplace(snode, AsyncState::Type::value);
        }
      }
    }
    if (auto global_atomic = stmt->cast<AtomicOpStmt>()) {
      if (auto ptr = global_atomic->dest->cast<GlobalPtrStmt>()) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_states.emplace(snode, AsyncState::Type::value);
          meta.output_states.emplace(snode, AsyncState::Type::value);
        }
      }
    }

    if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
      if (ptr->activate) {
        for (auto &snode : ptr->snodes.data) {
          meta.input_states.emplace(snode, AsyncState::Type::mask);
          meta.output_states.emplace(snode, AsyncState::Type::mask);
        }
      }
    }
    return false;
  });
  if (root_stmt->task_type == OffloadedStmt::listgen) {
    TI_ASSERT(root_stmt->snode->parent);
    meta.input_states.emplace(root_stmt->snode->parent, AsyncState::Type::list);
    meta.input_states.emplace(root_stmt->snode, AsyncState::Type::list);
    meta.input_states.emplace(root_stmt->snode, AsyncState::Type::mask);
    meta.output_states.emplace(root_stmt->snode, AsyncState::Type::list);
  } else if (root_stmt->task_type == OffloadedStmt::struct_for) {
    meta.input_states.emplace(root_stmt->snode, AsyncState::Type::list);
  }

  if (is_clear_list_task(root_stmt)) {
    // ClearList completely erases the element list, so its output list state
    // does NOT lead to a input state flow on the previous version of the list
    // state. However, a dependency edge (instead of flow edge) will still be
    // inserted since this is probably a WAR dependency.
    // TODO: or might be WAW? Do we even need to distinguish WAW and WAR?

    meta.output_states.emplace(get_snode_in_clear_list_task(root_stmt),
                               AsyncState::Type::list);
  }
  // TODO: this is probably not fully done. Hopefully after SFG Graphviz is
  // done we can easily spot what's left.
  return meta;
}

void AsyncEngine::enqueue(const TaskLaunchRecord &t) {
  if (offloaded_metas_.find(t.ir_handle) == offloaded_metas_.end()) {
    offloaded_metas_[t.ir_handle] = create_task_meta(t);
  }
  sfg->insert_task(t, offloaded_metas_[t.ir_handle]);
  task_queue.push_back(t);
}

void AsyncEngine::synchronize() {
  optimize_listgen();
  while (fuse())
    ;
  while (!task_queue.empty()) {
    queue.enqueue(task_queue.front());
    task_queue.pop_front();
  }
  queue.synchronize();
}

bool AsyncEngine::optimize_listgen() {
  // TODO: improve...
  bool modified = false;
  std::unordered_map<const SNode *, bool> list_dirty;
  auto new_task_queue = std::deque<TaskLaunchRecord>();
  for (int i = 0; i < task_queue.size(); i++) {
    // Try to eliminate unused listgens
    auto &t = task_queue[i];
    auto meta = offloaded_metas_[t.ir_handle];
    const auto *offload = t.stmt();
    bool keep = true;
    if (offload->task_type == OffloadedStmt::TaskType::listgen) {
      // keep
    } else if (is_clear_list_task(offload)) {
      // TODO: this only handles the case where the serial task contains exactly
      // one ClearListStmt. Shall we also handle fused cases?
      TI_ASSERT(task_queue[i + 1].stmt()->task_type ==
                OffloadedStmt::TaskType::listgen);
      const auto *snode = get_snode_in_clear_list_task(offload);
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
      new_task_queue.push_back(t);
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
      auto cloned_task_a = rec_a.ir_handle.clone();
      auto cloned_task_b = rec_b.ir_handle.clone();
      task_a = cloned_task_a->as<OffloadedStmt>();
      task_b = cloned_task_b->as<OffloadedStmt>();
      // TODO: in certain cases this optimization can be wrong!
      // Fuse task b into task_a
      for (int j = 0; j < (int)task_b->body->size(); j++) {
        task_a->body->insert(std::move(task_b->body->statements[j]));
      }
      task_b->body->statements.clear();

      // replace all reference to the offloaded statement B to A
      irpass::replace_all_usages_with(task_a, task_b, task_a);

      auto kernel = task_queue[i].kernel;
      irpass::full_simplify(task_a, /*after_lower_access=*/false, kernel);
      // For now, re_id is necessary for the hash to be correct.
      irpass::re_id(task_a);

      auto h = ir_bank_.get_hash(task_a);
      task_queue[i].ir_handle = IRHandle(task_a, h);
      ir_bank_.insert(std::move(cloned_task_a), h);
      task_queue[i + 1].ir_handle = IRHandle(nullptr, 0);

      // TODO: since cloned_task_b->body is empty, can we remove this (i.e.,
      //  simply delete cloned_task_b here)?
      ir_bank_.insert_to_trash_bin(std::move(cloned_task_b));

      modified = true;
      i++;  // skip fusing task_queue[i + 1] and task_queue[i + 2]
    }
  }

  auto new_task_queue = std::deque<TaskLaunchRecord>();

  // Eliminate empty tasks
  for (int i = 0; i < (int)task_queue.size(); i++) {
    if (task_queue[i].ir_handle.ir() != nullptr) {
      new_task_queue.push_back(task_queue[i]);
    }
  }

  task_queue = std::move(new_task_queue);

  return modified;
}

TLANG_NAMESPACE_END
