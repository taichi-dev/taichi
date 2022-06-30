#include "taichi/program/async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/system/timeline.h"
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/util/testing.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/extension.h"

// Keep this include in the end!
#include "taichi/program/async_profiler_switch.h"

TLANG_NAMESPACE_BEGIN

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

          auto config = kernel->program->config;
          auto ir = stmt;
          offload_to_executable(
              ir, config, kernel, /*verbose=*/false,
              /*determine_ad_stack_size=*/true,
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

AsyncEngine::AsyncEngine(const CompileConfig *const config,
                         const BackendExecCompilationFunc &compile_to_backend)
    : queue(&ir_bank_, compile_to_backend),
      config_(config),
      sfg(std::make_unique<StateFlowGraph>(this, &ir_bank_, config)) {
  Timeline::get_this_thread_instance().set_name("host");
  ir_bank_.set_sfg(sfg.get());
}

void AsyncEngine::launch(Kernel *kernel, RuntimeContext &context) {
  if (!kernel->lowered()) {
    kernel->lower(/*to_executable=*/false);
  }

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
  sfg->insert_tasks(records, config_->async_listgen_fast_filtering);
  if ((config_->async_flush_every > 0) &&
      (sfg->num_pending_tasks() >= config_->async_flush_every)) {
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
  if (config_->debug) {
    sfg->verify();
  }
  for (int pass = 0; pass < config_->async_opt_passes && modified; pass++) {
    modified = false;
    if (config_->async_opt_activation_demotion) {
      while (sfg->demote_activation()) {
        debug_sfg("act");
        modified = true;
      }
    }
    sfg->verify();
    if (config_->async_opt_listgen) {
      while (sfg->optimize_listgen()) {
        debug_sfg("listgen");
        modified = true;
      }
    }
    sfg->verify();
    if (config_->async_opt_dse) {
      while (sfg->optimize_dead_store()) {
        debug_sfg("dse");
        modified = true;
      }
    }
    sfg->verify();
    if (config_->async_opt_fusion) {
      auto max_iter = config_->async_opt_fusion_max_iter;
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
  auto prefix = config_->async_opt_intermediate_file;
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

  int return_code = std::system(
      fmt::format("dot -Tpdf -o {}.pdf {}.dot", dot_fn, dot_fn).c_str());
  if (return_code != 0) {
    throw std::runtime_error(
        fmt::format("Unable to convert {dot_fn}.dot into {dot_fn}.pdf")
            .c_str());
  }
}

TLANG_NAMESPACE_END
