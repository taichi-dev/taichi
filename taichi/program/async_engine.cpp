#include "async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

KernelLaunchRecord::KernelLaunchRecord(Context context, OffloadedStmt *stmt)
    : context(context), stmt(stmt) {
}

uint64 ExecutionQueue::hash(OffloadedStmt *stmt) {
  // TODO: upgrade this using IR comparisons
  std::string serialized;
  irpass::print(stmt, &serialized);
  uint64 ret = 0;
  for (uint64 i = 0; i < serialized.size(); i++) {
    ret = ret * 100000007UL + (uint64)serialized[i];
  }
  return ret;
}

void ExecutionQueue::enqueue(KernelLaunchRecord ker) {
  task_queue.push_back(ker);
}

void AsyncEngine::launch(taichi::lang::Kernel *kernel) {
  auto block = dynamic_cast<Block *>(kernel->ir);
  TI_ASSERT(block);
  auto &offloads = block->statements;
  for (std::size_t i = 0; i < offloads.size(); i++) {
    auto offload = offloads[i]->as<OffloadedStmt>();
    task_queue.emplace_back(kernel->program.context, offload);
  }
  optimize();
}

void AsyncEngine::synchronize() {
  while (!task_queue.empty()) {
    queue.enqueue(task_queue.front());
    task_queue.pop_front();
  }
  queue.synchronize();
}

TLANG_NAMESPACE_END
