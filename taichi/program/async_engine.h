#include <deque>
#include <thread>

#include "taichi/ir/ir.h"
#define TI_RUNTIME_HOST
#include "taichi/runtime/llvm/context.h"

TLANG_NAMESPACE_BEGIN

class KernelLaunchRecord {
 public:
  Context context;
  OffloadedStmt *stmt;

  KernelLaunchRecord(Context contxet, OffloadedStmt *stmt)
      : context(context), stmt(stmt) {
  }
};

// In charge of (parallel) compilation to binary and (serial) kernel launching
class ExecutionQueue {
 public:
  std::deque<KernelLaunchRecord> task_queue;

  std::vector<std::thread> compilation_workers;  // parallel
  std::thread launch_worker;                     // serial

  ExecutionQueue() {
  }

  void enqueue(KernelLaunchRecord ker) {
    task_queue.push_back(ker);
  }

  uint64 hash(OffloadedStmt *stmt) {
    // TODO: upgrade this using IR comparisons
    std::string serialized;
    irpass::print(stmt, &serialized);
    uint64 ret = 0;
    for (uint64 i = 0; i < serialized.size(); i++) {
      ret = ret * 100000007UL + (uint64)serialized[i];
    }
    return ret;
  }

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

  void launch(KernelLaunchRecord klr) {
    task_queue.push_back(klr);
    optimize();
  }

  void synchronize() {
    while (!task_queue.empty()) {
      Q.enqueue(task_queue.front());
      task_queue.pop_front();
    }
    Q.synchronize();
  }
};

TLANG_NAMESPACE_END
