#include "async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/common/testing.h"

TLANG_NAMESPACE_BEGIN

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

KernelLaunchRecord::KernelLaunchRecord(Context context,
                                       Kernel *kernel,
                                       OffloadedStmt *stmt)
    : context(context), kernel(kernel), stmt(stmt), h(hash(stmt)) {
}

void ExecutionQueue::enqueue(KernelLaunchRecord ker) {
  task_queue.push_back(ker);
}

void ExecutionQueue::synchronize() {
  TI_INFO("Flushing execution queue with {} tasks", task_queue.size());

  std::unordered_set<uint64> to_be_compiled;

  for (int i = 0; i < (int)task_queue.size(); i++) {
    auto ker = task_queue[i];
    auto h = ker.h;
    if (compiled_func.find(h) == compiled_func.end() &&
        to_be_compiled.find(h) == to_be_compiled.end()) {
      to_be_compiled.insert(h);
      compilation_workers.enqueue([&, ker, h, this]() {
        auto func = CodeGenCPU(ker.kernel, ker.stmt).codegen();
        std::lock_guard<std::mutex> _(mut);
        compiled_func[h] = func;
      });
    }
  }

  compilation_workers.flush();

  while (!task_queue.empty()) {
    auto ker = task_queue.front();
    launch_worker.enqueue([&, ker] {
      auto h = ker.h;
      FunctionType func;
      while (true) {
        std::unique_lock<std::mutex> lock(mut);
        if (compiled_func.find(h) == compiled_func.end()) {
          lock.unlock();
          Time::sleep(1e-6);
          continue;
        }
        func = compiled_func[h];
        break;
      }
      auto context = ker.context;
      func(context);
    });
    task_queue.pop_front();
  }

  auto t = Time::get_time();
  TI_TRACE("Flushing compilation & worker queue...");
  launch_worker.flush();
  TI_WARN("Flushing time {:.3f} ms", (Time::get_time() - t) * 1000);

  // uncomment for benchmarking
  // clear_cache();
}

ExecutionQueue::ExecutionQueue()
    : compilation_workers(4), launch_worker(1) {  // TODO: remove 4
}

void AsyncEngine::launch(Kernel *kernel) {
  if (!kernel->lowered)
    kernel->lower();
  auto block = dynamic_cast<Block *>(kernel->ir);
  TI_ASSERT(block);
  auto &offloads = block->statements;
  for (std::size_t i = 0; i < offloads.size(); i++) {
    auto offload = offloads[i]->as<OffloadedStmt>();
    task_queue.emplace_back(kernel->program.get_context(), kernel, offload);
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
