#include "taichi/program/async_engine.h"

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
  auto h = ker.h;
  if (compiled_func.find(h) == compiled_func.end() &&
      to_be_compiled.find(h) == to_be_compiled.end()) {
    to_be_compiled.insert(h);
    compilation_workers.enqueue([&, ker, h, this]() {
      {
        // Final lowering
        using namespace irpass;

        flag_access(ker.stmt);
        lower_access(ker.stmt, true, ker.kernel);
        flag_access(ker.stmt);
        full_simplify(ker.stmt, ker.kernel->program.config, ker.kernel);
        // analysis::verify(ker.stmt);
      }
      auto func = CodeGenCPU(ker.kernel, ker.stmt).codegen();
      std::lock_guard<std::mutex> _(mut);
      compiled_func[h] = func;
    });
  }

  launch_worker.enqueue([&, ker, h] {
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
}

void ExecutionQueue::synchronize() {
  TI_AUTO_PROF
  launch_worker.flush();
}

ExecutionQueue::ExecutionQueue()
    : compilation_workers(4), launch_worker(1) {  // TODO: remove 4
}

void AsyncEngine::launch(Kernel *kernel) {
  if (!kernel->lowered)
    kernel->lower(false);
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

struct TaskMeta {
  std::unordered_set<SNode *> input_snodes, output_snodes;
  std::unordered_set<SNode *> activation_snodes;
};

void AsyncEngine::optimize() {
  using namespace irpass::analysis;
  std::unordered_map<std::uint64_t, TaskMeta> metas;

  for (auto &t : task_queue) {
    auto &meta = metas[t.h];
    // TODO: this is an abuse...
    gather_statements(t.stmt, [&](Stmt *stmt) {
      if (auto global_ptr = stmt->cast<GlobalPtrStmt>()) {
        for (auto &snode : global_ptr->snodes.data) {
          meta.input_snodes.insert(snode);
        }
      }
      if (auto global_load = stmt->as<GlobalLoadStmt>()) {
        if (auto ptr = global_load->ptr->cast<GlobalPtrStmt>()) {
          for (auto &snode : ptr->snodes.data) {
            meta.input_snodes.insert(snode);
          }
        }
      }
      if (auto global_store = stmt->as<GlobalStoreStmt>()) {
        if (auto ptr = global_store->ptr->cast<GlobalPtrStmt>()) {
          for (auto &snode : ptr->snodes.data) {
            meta.output_snodes.insert(snode);
          }
        }
      }
      if (auto global_atomic = stmt->as<AtomicOpStmt>()) {
        if (auto ptr = global_atomic->dest->cast<GlobalPtrStmt>()) {
          for (auto &snode : ptr->snodes.data) {
            meta.input_snodes.insert(snode);
            meta.output_snodes.insert(snode);
          }
        }
      }

      if (auto ptr = stmt->as<GlobalPtrStmt>()) {
        if (ptr->activate) {
          for (auto &snode : ptr->snodes.data) {
            meta.activation_snodes.insert(snode);
          }
        }
      }
      return false;
    });
  }
}

TLANG_NAMESPACE_END
