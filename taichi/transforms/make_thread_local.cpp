#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

bool is_atomic_op_linear(AtomicOpType op_type) {
  return op_type == AtomicOpType::add || op_type == AtomicOpType ::sub;
}

void make_thread_local_offload(OffloadedStmt *offload) {
  irpass::print(offload);
  // TODO: deal with struct for
  if (offload->task_type != offload->range_for)
    return;

  // Gather all atomic adds/subs destinations
  // We do not use std::set to keep an deterministic order here.
  std::vector<GlobalPtrStmt *> atomic_destinations;
  // TODO: this is again an abuse since it gathers nothing. Need to design a IR
  // map/reduce system
  auto linear_atomics =
      irpass::analysis::gather_statements(offload, [&](Stmt *stmt) {
        if (auto atomic_op = stmt->cast<AtomicOpStmt>()) {
          if (is_atomic_op_linear(atomic_op->op_type)) {
            // Local atomics does not count
            if (auto dest = atomic_op->dest->cast<GlobalPtrStmt>()) {
              if (std::find(atomic_destinations.begin(),
                            atomic_destinations.end(),
                            dest) == atomic_destinations.end()) {
                atomic_destinations.push_back(dest);
              }
            }
          }
        }
        return false;
      });

  std::vector<GlobalPtrStmt *> ptr_to_reduce;

  for (auto dest : atomic_destinations) {
    // check if there is any other global load/store/atomic operations
    auto global_mem_ops =
        irpass::analysis::gather_statements(offload, [&](Stmt *stmt) {
          if (auto load = stmt->cast<GlobalLoadStmt>()) {
            if (maybe_same_address(load->ptr, dest)) {
              return true;
            }
          } else if (auto store = stmt->cast<GlobalStoreStmt>()) {
            if (maybe_same_address(store->ptr, dest)) {
              return true;
            }
          } else if (auto atomic = stmt->cast<AtomicOpStmt>()) {
            if (maybe_same_address(atomic->dest, dest)) {
              return !is_atomic_op_linear(atomic->op_type);
            }
          }
          return false;  // The statement is not related
        });
    TI_ASSERT(dest->width() == 1);
    // We can only optimized reductions to global ptrs with form like loss[None]
    // for now
    if (global_mem_ops.empty() && dest->snodes[0]->type == SNodeType::place &&
        dest->indices.empty()) {
      ptr_to_reduce.push_back(dest);
      TI_INFO("Detected reduction:");
      irpass::print(dest);
    }
  }

  TI_ASSERT(ptr_to_reduce.size() <= 1);
  for (auto ptr : ptr_to_reduce) {
    auto data_type = ptr->ret_type.data_type;
    auto offset = 0;
    // Step 1:
    // Create thread local storage
    {
      if (offload->prologue == nullptr) {
        offload->prologue = std::make_unique<Block>();
      }
      irpass::analysis::clone(ptr);
      auto tls_ptr = offload->prologue->push_back<ThreadLocalPtrStmt>(
          offset, VectorType(1, data_type));
      auto zero = offload->prologue->insert(
          std::make_unique<ConstStmt>(TypedConstant(data_type, 0)), -1);
      offload->prologue->push_back<GlobalStoreStmt>(tls_ptr, zero);
    }

    // Step 2:
    // Make loop body accumulate to TLS
    {
      auto tls_ptr = offload->body->insert(
          Stmt::make<ThreadLocalPtrStmt>(offset, VectorType(1, data_type)), 0);
      ptr->replace_with(tls_ptr);
    }

    // Step 3:
    // Atomic-add contribution to global version
    {
      if (offload->epilogue == nullptr) {
        offload->epilogue = std::make_unique<Block>();
      }
      auto tls_ptr = offload->epilogue->push_back<ThreadLocalPtrStmt>(
          offset, VectorType(1, data_type));
      // TODO: do not use global load from TLS.
      auto tls_load = offload->epilogue->push_back<GlobalLoadStmt>(tls_ptr);
      auto global_ptr = offload->epilogue->insert(
          std::unique_ptr<Stmt>((Stmt *)irpass::analysis::clone(ptr).release()),
          -1);
      offload->epilogue->push_back<AtomicOpStmt>(AtomicOpType::add, global_ptr,
                                                 tls_load);
    }
  }
}

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_thread_local(IRNode *root) {
  TI_AUTO_PROF;
  auto root_block = root->cast<Block>();
  TI_ASSERT(root_block);
  for (auto &offload : root_block->statements) {
    make_thread_local_offload(offload->cast<OffloadedStmt>());
  }
  irpass::typecheck(root);
  fix_block_parents(root);
  TI_INFO("after:");
  irpass::print(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
