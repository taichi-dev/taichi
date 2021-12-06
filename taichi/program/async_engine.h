#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "taichi/ir/ir.h"
#include "taichi/lang_util.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST
#include "taichi/program/async_utils.h"
#include "taichi/program/ir_bank.h"
#include "taichi/program/state_flow_graph.h"

TLANG_NAMESPACE_BEGIN

// TODO(yuanming-hu): split into multiple files

class ParallelExecutor {
 public:
  using TaskType = std::function<void()>;

  explicit ParallelExecutor(const std::string &name, int num_threads);
  ~ParallelExecutor();

  void enqueue(const TaskType &func);

  void flush();

  int get_num_threads() {
    return num_threads_;
  }

 private:
  enum class ExecutorStatus {
    uninitialized,
    initialized,
    finalized,
  };

  void worker_loop();

  // Must be called while holding |mut|.
  bool flush_cv_cond();

  std::string name_;
  int num_threads_;
  std::atomic<int> thread_counter_{0};
  std::mutex mut_;

  // All guarded by |mut|
  ExecutorStatus status_;
  std::vector<std::thread> threads_;
  std::deque<TaskType> task_queue_;
  int running_threads_;

  // Used to signal the workers that they can start polling from |task_queue|.
  std::condition_variable init_cv_;
  // Used by |this| to instruct the worker thread that there is an event:
  // * task being enqueued
  // * shutting down
  std::condition_variable worker_cv_;
  // Used by a worker thread to unblock the caller from waiting for a flush.
  //
  // TODO: Instead of having this as a member variable, we can enqueue a
  // callback upon flush(). The flush() will then block waiting for that
  // callback to be executed?
  std::condition_variable flush_cv_;
};

// Compiles the offloaded and optimized IR to the target backend's executable.
using BackendExecCompilationFunc =
    std::function<FunctionType(Kernel &, OffloadedStmt *)>;

// In charge of (parallel) compilation to binary and (serial) kernel launching
class ExecutionQueue {
 public:
  std::mutex mut;

  ParallelExecutor compilation_workers;  // parallel compilation
  ParallelExecutor launch_worker;        // serial launching

  explicit ExecutionQueue(IRBank *ir_bank,
                          const BackendExecCompilationFunc &compile_to_backend);

  void enqueue(const TaskLaunchRecord &ker);

  void compile_task() {
  }

  void launch_task() {
  }

  void clear_cache() {
    compiled_funcs_.clear();
  }

  void synchronize();

 private:
  // Wraps an executable function that is compiled from a task asynchronously.
  class AsyncCompiledFunc {
   public:
    AsyncCompiledFunc() : f_(p_.get_future()) {
    }

    inline void set(const FunctionType &func) {
      p_.set_value(func);
    }

    inline FunctionType get() {
      return f_.get();
    }

   private:
    std::promise<FunctionType> p_;
    // https://stackoverflow.com/questions/38160960/calling-stdfutureget-repeatedly
    std::shared_future<FunctionType> f_;
  };
  std::unordered_map<uint64, AsyncCompiledFunc> compiled_funcs_;

  IRBank *ir_bank_;  // not owned
  BackendExecCompilationFunc compile_to_backend_;
};

// An engine for asynchronous execution and optimization
class AsyncEngine {
 public:
  // TODO: state machine

  ExecutionQueue queue;
  const CompileConfig *const config_;

  std::unique_ptr<StateFlowGraph> sfg;

  explicit AsyncEngine(const CompileConfig *const config,
                       const BackendExecCompilationFunc &compile_to_backend);

  void clear_cache() {
    queue.clear_cache();
  }

  void launch(Kernel *kernel, RuntimeContext &context);

  // Flush the tasks only.
  void flush();
  // Flush the tasks and block waiting for the GPU device to complete.
  void synchronize();

  void debug_sfg(const std::string &suffix);

 private:
  IRBank ir_bank_;

  struct KernelMeta {
    // OffloadedCachedData holds some data that needs to be computed once for
    // each offloaded task of a kernel. Especially, it holds a cloned offloaded
    // task, but uses it as a READ-ONLY template. That is, code that later finds
    // it necessary to mutate this task (e.g. kernel fusion) should do another
    // clone, so that the template in this class stays untouched.
    //
    // This design allows us to do task cloning lazily. It turned out that doing
    // clone on every kernel launch is too expensive.
    std::vector<IRHandle> ir_handle_cached;
  };

  std::unordered_map<const Kernel *, KernelMeta> kernel_metas_;
  // How many times we have flushed
  int flush_counter_{0};
  // How many times we have synchronized
  int sync_counter_{0};
  int cur_sync_sfg_debug_counter_{0};
  std::unordered_map<std::string, int> cur_sync_sfg_debug_per_stage_counts_;
};

TLANG_NAMESPACE_END
