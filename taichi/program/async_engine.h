#include <deque>
#include <thread>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/runtime/llvm/context.h"
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

// TODO(yuanming-hu): split into multiple files

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
