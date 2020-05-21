#include <deque>
#include <thread>
#include <mutex>
#include <atomic>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/runtime/llvm/context.h"
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

// TODO(yuanming-hu): split into multiple files

// TODO: use semaphores instead of Time::sleep
class ParallelExecutor {
 public:
  using TaskType = std::function<void()>;

  explicit ParallelExecutor(int num_threads)
      : num_threads(num_threads),
        status(ExecutorStatus::uninitialized),
        running_threads(0) {
    auto _ = std::lock_guard<std::mutex>(mut);

    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back([this]() { this->task(); });
    }

    status = ExecutorStatus::initialized;
  }

  void enqueue(const TaskType &func) {
    std::lock_guard<std::mutex> _(mut);
    task_queue.push_back(func);
  }

  void flush() {
    while (true) {
      std::unique_lock<std::mutex> lock(mut);
      if (task_queue.empty() && running_threads == 0) {
        break;
      } else {
        lock.unlock();
        Time::sleep(1e-6);
      }
    }
  }

  ~ParallelExecutor() {
    flush();
    {
      auto _ = std::lock_guard<std::mutex>(mut);
      status = ExecutorStatus::finalized;
    }
    for (auto &th : threads) {
      th.join();
    }
  }

  int get_num_threads() {
    return num_threads;
  }

 private:
  enum class ExecutorStatus {
    uninitialized,
    initialized,
    finalized,
  };

  void task() {
    TI_DEBUG("Starting worker thread.");
    while (true) {
      std::unique_lock<std::mutex> lock(mut);
      if (status == ExecutorStatus::uninitialized) {
        lock.unlock();
        Time::sleep(1e-6);
        continue;  // wait until initialized
      }
      if (status == ExecutorStatus::finalized && task_queue.empty()) {
        break;  // finalized, exit
      }
      // initialized and not finalized. Do work.
      if (!task_queue.empty()) {
        auto task = task_queue.front();
        running_threads++;
        task_queue.pop_front();
        lock.unlock();
        // Run the task
        task();
        running_threads--;
      }
    }
  }

  int num_threads;
  std::mutex mut;
  ExecutorStatus status;

  std::vector<std::thread> threads;
  std::deque<TaskType> task_queue;
  std::atomic<int> running_threads;
};

class KernelLaunchRecord {
 public:
  Context context;
  Kernel *kernel;  // TODO: remove this
  OffloadedStmt *stmt;
  std::unique_ptr<IRNode> stmt_;
  uint64 h;

  KernelLaunchRecord(Context contxet,
                     Kernel *kernel,
                     std::unique_ptr<IRNode> &&stmt);
};

// In charge of (parallel) compilation to binary and (serial) kernel launching
class ExecutionQueue {
 public:
  std::mutex mut;
  std::deque<KernelLaunchRecord> task_queue;
  std::vector<KernelLaunchRecord> trashbin;  // prevent IR from being deleted
  std::unordered_set<uint64> to_be_compiled;

  ParallelExecutor compilation_workers;  // parallel compilation
  ParallelExecutor launch_worker;        // serial launching

  std::unordered_map<uint64, FunctionType> compiled_func;

  ExecutionQueue();

  void enqueue(KernelLaunchRecord &&ker);

  void compile_task() {
  }

  void launch_task() {
  }

  void clear_cache() {
    compiled_func.clear();
  }

  void synchronize();
};

// An engine for asynchronous execution and optimization

class AsyncEngine {
 public:
  // TODO: state machine

  struct TaskMeta {
    std::unordered_set<SNode *> input_snodes, output_snodes;
    std::unordered_set<SNode *> activation_snodes;
  };

  std::unordered_map<std::uint64_t, TaskMeta> metas;

  ExecutionQueue queue;

  std::deque<KernelLaunchRecord> task_queue;

  AsyncEngine() {
  }

  bool optimize_listgen();  // return true when modified

  bool fuse();  // return true when modified

  void clear_cache() {
    queue.clear_cache();
  }

  void launch(Kernel *kernel);

  void enqueue(KernelLaunchRecord &&t);

  void synchronize();
};

TLANG_NAMESPACE_END
