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
          TI_TAG;
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
}

}  // namespace irpass

TLANG_NAMESPACE_END
