#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/lang_util.h"
#include "taichi/program/context.h"

TLANG_NAMESPACE_BEGIN

// TODO(yuanming-hu): split into multiple files

class IRHandle {
 public:
  IRHandle(IRNode const *ir, uint64 hash) : ir_(ir), hash_(hash) {
  }

  std::unique_ptr<IRNode> clone() const;

  // Set this->ir_ to the cloned one.
  std::unique_ptr<IRNode> clone_on_write();

  IRNode const *ir() {
    return ir_;
  }

  uint64 hash() const {
    return hash_;
  }

  // Two IRHandles are considered the same if their hash values are the same.
  bool operator==(const IRHandle &other_ir_handle) const {
    return hash_ == other_ir_handle.hash_;
  }

 private:
  IRNode const *ir_;  // not owned
  uint64 hash_;
};

TLANG_NAMESPACE_END

namespace std {
template <>
struct hash<taichi::lang::IRHandle> {
  std::size_t operator()(taichi::lang::IRHandle const &ir_handle) const
      noexcept {
    return ir_handle.hash();
  }
};
}  // namespace std

TLANG_NAMESPACE_BEGIN

class IRBank {
 public:
  std::vector<std::unique_ptr<IRNode>> trash_bin;

  uint64 get_hash(IRNode *ir);
  void insert(std::unique_ptr<IRNode> &&ir, uint64 hash);
//  void insert(std::unique_ptr<IRNode> &&ir);
  IRNode *find(IRHandle ir_handle);

 private:
  std::unordered_map<IRNode *, uint64> hash_bank_;
  std::unordered_map<IRHandle, std::unique_ptr<IRNode>> ir_bank_;
  //  std::unordered_map<std::pair<IRHandle, IRHandle>, IRHandle> fuse_bank_;
};

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

  // Must be called while holding |mut|.
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

// Records the necessary data for launching an offloaded task.
class TaskLaunchRecord {
 public:
  Context context;
  Kernel *kernel;  // TODO: remove this
  IRHandle ir_handle;

  TaskLaunchRecord(Context context, Kernel *kernel, IRHandle ir_handle);

  inline OffloadedStmt *stmt() {
    return const_cast<IRNode *>(ir_handle.ir())->as<OffloadedStmt>();
  }
};

// In charge of (parallel) compilation to binary and (serial) kernel launching
class ExecutionQueue {
 public:
  std::mutex mut;
  std::deque<TaskLaunchRecord> task_queue;

  // prevent IR from being deleted
  std::vector<std::unique_ptr<IRNode>> trash_bin;

  ParallelExecutor compilation_workers;  // parallel compilation
  ParallelExecutor launch_worker;        // serial launching

  ExecutionQueue();

  void enqueue(TaskLaunchRecord &&ker);

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
};

// An engine for asynchronous execution and optimization
class AsyncEngine {
 public:
  // TODO: state machine

  ExecutionQueue queue;

  std::deque<TaskLaunchRecord> task_queue;

  AsyncEngine() {
  }

  bool optimize_listgen();  // return true when modified

  bool fuse();  // return true when modified

  void clear_cache() {
    queue.clear_cache();
  }

  void launch(Kernel *kernel, Context &context);

  void enqueue(TaskLaunchRecord &&t);

  void synchronize();

 private:
  IRBank ir_bank_;
  // KernelMeta is unused now.
  struct KernelMeta {
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

    bool initialized{false};
  };

  struct TaskMeta {
    std::unordered_set<SNode *> input_snodes, output_snodes;
    std::unordered_set<SNode *> activation_snodes;
  };

  // In async mode, the root of an AST is an OffloadedStmt instead of a Block.
  // This map provides a dummy Block root for these OffloadedStmt, so that
  // get_kernel() could still work correctly.
  std::unordered_map<const Kernel *, KernelMeta> kernel_metas_;
  std::unordered_map<IRHandle, TaskMeta> offloaded_metas_;
};

TLANG_NAMESPACE_END
