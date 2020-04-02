#include <deque>
#include <thread>

#include "taichi/ir/ir.h"
#define TI_RUNTIME_HOST
#include "taichi/runtime/llvm/context.h"

TLANG_NAMESPACE_BEGIN

// TODO(yuanming-hu): split into multiple files

class KernelLaunchRecord {
 public:
  Context context;
  OffloadedStmt *stmt;

  KernelLaunchRecord(Context contxet, OffloadedStmt *stmt);
};

// In charge of (parallel) compilation to binary and (serial) kernel launching
class ExecutionQueue {
 public:
  std::deque<KernelLaunchRecord> task_queue;

  std::vector<std::thread> compilation_workers;  // parallel
  std::thread launch_worker;                     // serial

  ExecutionQueue() {
  }

  void enqueue(KernelLaunchRecord ker);

  uint64 hash(OffloadedStmt *stmt);

  void compile_task() {
  }

  void launch_task() {
  }

  void synchronize() {
  }
};

// An engine for asynchronous execution and optimization

class AsyncEngine {
 public:
  // TODO: state machine

  ExecutionQueue Q;

  std::deque<KernelLaunchRecord> task_queue;

  AsyncEngine() {
  }

  void optimize() {
  }

  void launch(Kernel *kernel);

  void synchronize();
};

TLANG_NAMESPACE_END
