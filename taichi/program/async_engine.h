#include <atomic>
#include <condition_variable>
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
class ParallelExecutor {
 public:
  using TaskType = std::function<void()>;

  explicit ParallelExecutor(int num_threads);
  ~ParallelExecutor();

  void enqueue(const TaskType &func);

  void flush();

  int get_num_threads() {
    return num_threads;
  }

 private:
  enum class ExecutorStatus {
    uninitialized,
    initialized,
    finalized,
  };

  void worker_loop();

  // Must be called whil holding |mut|.
  bool flush_cv_cond();

  int num_threads;
  std::mutex mut;

  // All guarded by |mut|
  ExecutorStatus status;
  std::vector<std::thread> threads;
  std::deque<TaskType> task_queue;
  int running_threads;

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
