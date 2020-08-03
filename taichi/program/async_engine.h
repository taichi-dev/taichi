#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/lang_util.h"
#include "taichi/runtime/llvm/context.h"

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
  uint64 h;        // hash of |stmt|

  KernelLaunchRecord(Context context,
                     Kernel *kernel,
                     OffloadedStmt *stmt,
                     uint64 h,
                     Block *dummy_root);

  inline OffloadedStmt *stmt() {
    return stmt_;
  }

  // When we need to make changes to |stmt|, call this method so that the |stmt|
  // is cloned from the template, so that the template itself remains untouched.
  //
  // Cloning will only happen on the first call.
  OffloadedStmt *clone_stmt_on_write();

 private:
  // This begins as the template in OffloadedCachedData. If
  // clone_stmt_on_write() is invoked, it points to the underlying pointer owned
  // by |cloned_stmt_holder_|.
  OffloadedStmt *stmt_;

  // These are for cloning |stmt_|.
  Block *dummy_root_;  // Not owned
  std::unique_ptr<OffloadedStmt> cloned_stmt_holder_;
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

 private:
  struct KernelMeta {
    std::unique_ptr<Block> dummy_root;

    // OffloadedCachedData holds some data that needs to be computed once for
    // each offloaded task of a kernel. Especially, it holds a cloned offloaded
    // task, but uses it as a READ-ONLY template. That is, code that later finds
    // it necessary to mutate this task (e.g. kernel fusion) should do another
    // clone, so that the template in this class stays untouched.
    //
    // This design allows us to do task cloning lazily. It turned out that doing
    // clone on every kernel launch is too expensive.
    struct OffloadedCachedData {
     public:
      explicit OffloadedCachedData(std::unique_ptr<OffloadedStmt> &&tmpl,
                                   uint64 hash)
          : tmpl_(std::move(tmpl)), hash_(hash) {
      }

      // Get the read-only offloaded task template. Ideally this should be a
      // const pointer, but the IR passes won't work...
      inline OffloadedStmt *get_template() {
        return tmpl_.get();
      }

      inline uint64 get_hash() const {
        return hash_;
      }

     private:
      // Hide the unique pointer so that the ownership cannot be accidentally
      // transferred.
      std::unique_ptr<OffloadedStmt> tmpl_;
      uint64 hash_;
    };

    std::vector<OffloadedCachedData> offloaded_cached;

    inline bool initialized() const {
      return dummy_root != nullptr;
    }
  };

  struct TaskMeta {
    std::unordered_set<SNode *> input_snodes, output_snodes;
    std::unordered_set<SNode *> activation_snodes;
  };

  // In async mode, the root of an AST is an OffloadedStmt instead of a Block.
  // This map provides a dummy Block root for these OffloadedStmt, so that
  // get_kernel() could still work correctly.
  std::unordered_map<const Kernel *, KernelMeta> kernel_metas_;
  std::unordered_map<std::uint64_t, TaskMeta> offloaded_metas_;
};

TLANG_NAMESPACE_END
