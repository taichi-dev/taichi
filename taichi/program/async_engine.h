#include <deque>
#include <thread>
#include <mutex>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/runtime/llvm/context.h"
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

// TODO(yuanming-hu): split into multiple files

// TODO: use semaphores instead of Time::sleep
class ParallelExecutor {
 public:
  using TaskType = std::function<void()>;

  enum class ExecutorStatus {
    uninitialized,
    initialized,
    finalized,
  };

  explicit ParallelExecutor(int num_threads)
      : num_threads(num_threads), status(ExecutorStatus::uninitialized) {
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
      if (task_queue.empty()) {
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
  void task() {
    TI_DEBUG("Starting worker thread.");
    while (true) {
      std::unique_lock<std::mutex> lock(mut);
      if (status == ExecutorStatus::uninitialized) {
        lock.unlock();
        Time::sleep(1e-6);
        continue;  // wait until initialized
      }
      if (status == ExecutorStatus::finalized) {
        break;  // finalized, exit
      }
      // initialized and not finalized. Do work.
      if (!task_queue.empty()) {
        auto task = task_queue.front();
        task_queue.pop_front();
        lock.unlock();
        // Run the task
        task();
      }
    }
    TI_DEBUG("Exiting worker thread.");
  }

  int num_threads;
  std::mutex mut;
  ExecutorStatus status;

  std::vector<std::thread> threads;
  std::deque<TaskType> task_queue;
};

class KernelLaunchRecord {
 public:
  Context context;
  Kernel *kernel;  // TODO: remove this
  OffloadedStmt *stmt;

  KernelLaunchRecord(Context contxet, Kernel *kernel, OffloadedStmt *stmt);
};

// In charge of (parallel) compilation to binary and (serial) kernel launching
class ExecutionQueue {
 public:
  std::deque<KernelLaunchRecord> task_queue;

  std::vector<std::thread> compilation_workers;  // parallel
  std::thread launch_worker;                     // serial

  std::unordered_map<uint64, FunctionType> compiled_func;

  ExecutionQueue() {
  }

  void enqueue(KernelLaunchRecord ker);

  uint64 hash(OffloadedStmt *stmt);

  void compile_task() {
  }

  void launch_task() {
  }

  void synchronize();
};

// An engine for asynchronous execution and optimization

class AsyncEngine {
 public:
  // TODO: state machine

  ExecutionQueue queue;

  std::deque<KernelLaunchRecord> task_queue;

  AsyncEngine() {
  }

  void optimize() {
  }

  void launch(Kernel *kernel);

  void synchronize();
};

TLANG_NAMESPACE_END
