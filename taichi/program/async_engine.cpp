#include "taichi/program/async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/system/timeline.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/util/testing.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/extension.h"

// Keep this include in the end!
#include "taichi/program/async_profiler_switch.h"

TLANG_NAMESPACE_BEGIN

ParallelExecutor::ParallelExecutor(const std::string &name, int num_threads)
    : name_(name),
      num_threads(num_threads),
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
  auto thread_id = thread_counter++;

  std::string thread_name = name_;
  if (num_threads != 1)
    thread_name += fmt::format("_{}", thread_id);
  Timeline::get_this_thread_instance().set_name(thread_name);

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
  // TODO: for now we are using kernel name for task name. It may be helpful to
  // use the real task name.
  auto kernel_name = kernel->name;

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

    compilation_workers.enqueue(
        [kernel_name, async_func, stmt, kernel, this]() {
          TI_TIMELINE(kernel_name);
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
          auto func = this->compile_to_backend_(*kernel, stmt);
          async_func->set(func);
        });
    ir_bank_->insert_to_trash_bin(std::move(cloned_stmt));
  }

  launch_worker.enqueue(
      [kernel_name, async_func, context = ker.context]() mutable {
        TI_TIMELINE(kernel_name);
        auto func = async_func->get();
        func(context);
      });
}

void ExecutionQueue::synchronize() {
  TI_AUTO_PROF;
  launch_worker.flush();
}

ExecutionQueue::ExecutionQueue(
    IRBank *ir_bank,
    const BackendExecCompilationFunc &compile_to_backend)
    : compilation_workers("compiler", 4),  // TODO: remove 4
      launch_worker("launcher", 1),
      ir_bank_(ir_bank),
      compile_to_backend_(compile_to_backend) {
}

AsyncEngine::AsyncEngine(Program *program,
                         const BackendExecCompilationFunc &compile_to_backend)
    : queue(&ir_bank_, compile_to_backend),
      program(program),
      sfg(std::make_unique<StateFlowGraph>(this, &ir_bank_)) {
  Timeline::get_this_thread_instance().set_name("host");
  ir_bank_.set_sfg(sfg.get());
}

void AsyncEngine::launch(Kernel *kernel, Context &context) {
  if (!kernel->lowered)
    kernel->lower(/*to_executable=*/false);

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  TI_ASSERT(block);

  auto &offloads = block->statements;
  auto &kmeta = kernel_metas_[kernel];
  const bool kmeta_inited = !kmeta.ir_handle_cached.empty();
  std::vector<TaskLaunchRecord> records;
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
    records.push_back(rec);
  }
  const auto &config = program->config;
  sfg->insert_tasks(records, config.async_listgen_fast_filtering);
  if ((config.async_flush_every > 0) &&
      (sfg->num_pending_tasks() >= config.async_flush_every)) {
    TI_TRACE("Async flushing {} tasks", sfg->num_pending_tasks());
    flush();
  }
}

void AsyncEngine::synchronize() {
  TI_AUTO_PROF;
  flush();
  queue.synchronize();

  sync_counter_++;
  // Clear SFG debug stats
  cur_sync_sfg_debug_counter_ = 0;
  cur_sync_sfg_debug_per_stage_counts_.clear();
}

void AsyncEngine::flush() {
  TI_AUTO_PROF;
  TI_AUTO_TIMELINE;

  bool modified = true;
  sfg->reid_nodes();
  sfg->reid_pending_nodes();
  sfg->sort_node_edges();
  TI_TRACE("Synchronizing SFG of {} nodes ({} pending)", sfg->size(),
           sfg->num_pending_tasks());
  debug_sfg("initial");
  if (program->config.debug) {
    sfg->verify();
  }
  for (int pass = 0; pass < program->config.async_opt_passes && modified;
       pass++) {
    modified = false;
    if (program->config.async_opt_activation_demotion) {
      while (sfg->demote_activation()) {
        debug_sfg("act");
        modified = true;
      }
    }
    sfg->verify();
    if (program->config.async_opt_listgen) {
      while (sfg->optimize_listgen()) {
        debug_sfg("listgen");
        modified = true;
      }
    }
    sfg->verify();
    if (program->config.async_opt_dse) {
      while (sfg->optimize_dead_store()) {
        debug_sfg("dse");
        modified = true;
      }
    }
    sfg->verify();
    if (program->config.async_opt_fusion) {
      auto max_iter = program->config.async_opt_fusion_max_iter;
      for (int iter = 0; max_iter == 0 || iter < max_iter; iter++) {
        if (sfg->fuse()) {
          debug_sfg("fuse");
          modified = true;
        } else {
          break;
        }
      }
    }
    sfg->verify();
  }
  debug_sfg("final");
  {
    TI_TIMELINE("enqueue");
    auto tasks = sfg->extract_to_execute();
    TI_TRACE("Ended up with {} nodes", tasks.size());
    for (auto &task : tasks) {
      queue.enqueue(task);
    }
  }
  flush_counter_++;
}

void AsyncEngine::debug_sfg(const std::string &stage) {
  TI_TRACE("Ran {}, counter={}", stage, cur_sync_sfg_debug_counter_);
  auto prefix = program->config.async_opt_intermediate_file;
  if (prefix.empty())
    return;
  auto dot = sfg->dump_dot(/*rankdir=*/std::nullopt);
  constexpr int debug_limit = 100;
  if (cur_sync_sfg_debug_counter_ >= debug_limit) {
    TI_WARN("Too many (> {}) debug outputs. debug_sfg invocation Ignored.",
            debug_limit);
    return;
  }
  auto dot_fn =
      fmt::format("{}_flush{:04d}_sync{:04d}_{:04d}_{}", prefix, flush_counter_,
                  sync_counter_, cur_sync_sfg_debug_counter_++, stage);
  auto stage_count = cur_sync_sfg_debug_per_stage_counts_[stage]++;
  if (stage_count) {
    dot_fn += std::to_string(stage_count);
  }
  {
    std::ofstream dot_file(dot_fn + ".dot");
    dot_file << dot;
  }
  std::system(
      fmt::format("dot -Tpdf -o {}.pdf {}.dot", dot_fn, dot_fn).c_str());
}

TLANG_NAMESPACE_END
