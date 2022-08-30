#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

namespace {

// Find the destinations of global atomic reductions that can be demoted into
// TLS buffer.
template <typename T>
std::vector<std::pair<T *, AtomicOpType>> find_global_reduction_destinations(
    OffloadedStmt *offload,
    const std::function<bool(T *)> &dest_checker) {
  static_assert(std::is_same_v<T, GlobalPtrStmt> ||
                std::is_same_v<T, GlobalTemporaryStmt>);
  // Gather all atomic add/sub/max/min destinations and record corresponding op
  // type on the first appearance of a destination.
  // Only one op type will be allowed on one destination (add/sub is an
  // exception because add/sub can be mixed together in delta calculation).
  // We use std::vector instead of std::map to keep a deterministic order here.
  std::vector<std::pair<T *, AtomicOpType>> atomic_destinations;
  // TODO: this is again an abuse since it gathers nothing. Need to design a IR
  // map/reduce system
  auto atomics = irpass::analysis::gather_statements(offload, [&](Stmt *stmt) {
    if (auto atomic_op = stmt->cast<AtomicOpStmt>()) {
      if (atomic_op->op_type == AtomicOpType::add ||
          atomic_op->op_type == AtomicOpType::sub ||
          atomic_op->op_type == AtomicOpType::max ||
          atomic_op->op_type == AtomicOpType::min) {
        // Local atomics do not count.
        if (auto dest = atomic_op->dest->cast<T>()) {
          if (std::find_if(atomic_destinations.begin(),
                           atomic_destinations.end(),
                           [&](const std::pair<T *, AtomicOpType> &elem) {
                             return elem.first == dest;
                           }) == atomic_destinations.end()) {
            atomic_destinations.push_back(
                {dest, atomic_op->op_type == AtomicOpType::sub
                           ? AtomicOpType::add
                           : atomic_op->op_type});
          }
        }
      }
    }
    return false;
  });

  std::vector<std::pair<T *, AtomicOpType>> valid_reduction_values;
  for (auto dest : atomic_destinations) {
    // check if there is any other global load/store/atomic operations
    auto related_global_mem_ops =
        irpass::analysis::gather_statements(offload, [&](Stmt *stmt) {
          if (auto load = stmt->cast<GlobalLoadStmt>()) {
            if (irpass::analysis::maybe_same_address(load->src, dest.first)) {
              return true;
            }
          } else if (auto store = stmt->cast<GlobalStoreStmt>()) {
            if (irpass::analysis::maybe_same_address(store->dest, dest.first)) {
              return true;
            }
          } else if (auto atomic = stmt->cast<AtomicOpStmt>()) {
            if (irpass::analysis::maybe_same_address(atomic->dest,
                                                     dest.first)) {
              return !((atomic->op_type == AtomicOpType::sub &&
                        dest.second == AtomicOpType::add) ||
                       atomic->op_type == dest.second);
            }
          }
          for (auto &op : stmt->get_operands()) {
            // Make sure the values of related atomic operations are not used.
            if (auto atomic = op->cast<AtomicOpStmt>()) {
              if (irpass::analysis::maybe_same_address(atomic->dest,
                                                       dest.first)) {
                return true;
              }
            }
          }
          return false;  // Now we are sure the statement is not related to the
                         // destination
        });
    if (related_global_mem_ops.empty() && dest_checker(dest.first)) {
      valid_reduction_values.push_back(dest);
    }
  }
  return valid_reduction_values;
}

void make_thread_local_offload(OffloadedStmt *offload) {
  if (offload->task_type != OffloadedTaskType::range_for &&
      offload->task_type != OffloadedTaskType::struct_for)
    return;

  std::vector<std::pair<Stmt *, AtomicOpType>> valid_reduction_values;
  {
    auto valid_global_ptrs = find_global_reduction_destinations<GlobalPtrStmt>(
        offload, [](GlobalPtrStmt *dest) {
          // We can only optimized reductions to global ptrs with form like
          // loss[None] (0-D fields) for now.
          // No TLS on quant types.
          return (dest->snode->type == SNodeType::place) &&
                 dest->indices.empty() && dest->snode->dt->is<PrimitiveType>();
        });
    auto valid_global_tmps =
        find_global_reduction_destinations<GlobalTemporaryStmt>(
            offload, [](auto *) { return true; });
    std::copy(valid_global_ptrs.begin(), valid_global_ptrs.end(),
              std::back_inserter(valid_reduction_values));
    std::copy(valid_global_tmps.begin(), valid_global_tmps.end(),
              std::back_inserter(valid_reduction_values));
  }

  std::size_t tls_offset = 0;

  // TODO: sort thread local storage variables according to dtype_size to
  // reduce buffer fragmentation.
  for (auto dest : valid_reduction_values) {
    auto data_type = dest.first->ret_type.ptr_removed();
    auto dtype_size = data_type_size(data_type);
    // Step 1:
    // Create thread local storage
    {
      if (offload->tls_prologue == nullptr) {
        offload->tls_prologue = std::make_unique<Block>();
        offload->tls_prologue->parent_stmt = offload;
      }

      // ensure alignment
      tls_offset += (dtype_size - tls_offset % dtype_size) % dtype_size;

      auto tls_ptr = offload->tls_prologue->push_back<ThreadLocalPtrStmt>(
          tls_offset, TypeFactory::get_instance().get_pointer_type(data_type));

      auto zero = offload->tls_prologue->insert(
          std::make_unique<ConstStmt>(
              dest.second == AtomicOpType::max   ? get_min_value(data_type)
              : dest.second == AtomicOpType::min ? get_max_value(data_type)
                                                 : TypedConstant(data_type, 0)),
          -1);
      // Zero-fill
      // TODO: do not use GlobalStore for TLS ptr.
      offload->tls_prologue->push_back<GlobalStoreStmt>(tls_ptr, zero);
    }

    // Step 2:
    // Make loop body accumulate to TLS ptr instead of global ptr
    {
      auto tls_ptr = offload->body->insert(
          Stmt::make<ThreadLocalPtrStmt>(
              tls_offset,
              TypeFactory::get_instance().get_pointer_type(data_type)),
          0);
      dest.first->replace_usages_with(tls_ptr);
    }

    // Step 3:
    // Atomic-add thread local contribution to its global version
    {
      if (offload->tls_epilogue == nullptr) {
        offload->tls_epilogue = std::make_unique<Block>();
        offload->tls_epilogue->parent_stmt = offload;
      }
      auto tls_ptr = offload->tls_epilogue->push_back<ThreadLocalPtrStmt>(
          tls_offset, TypeFactory::get_instance().get_pointer_type(data_type));
      // TODO: do not use global load from TLS.
      auto tls_load = offload->tls_epilogue->push_back<GlobalLoadStmt>(tls_ptr);
      auto global_ptr = offload->tls_epilogue->insert(
          std::unique_ptr<Stmt>(
              (Stmt *)irpass::analysis::clone(dest.first).release()),
          -1);
      offload->tls_epilogue->insert(
          AtomicOpStmt::make_for_reduction(dest.second, global_ptr, tls_load),
          -1);
    }

    // allocate storage for the TLS variable
    tls_offset += dtype_size;
  }

  offload->tls_size = std::max(std::size_t(1), tls_offset);
}

}  // namespace

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_thread_local(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      make_thread_local_offload(offload->cast<OffloadedStmt>());
    }
  } else {
    make_thread_local_offload(root->as<OffloadedStmt>());
  }
  type_check(root, config);
}

}  // namespace irpass

TLANG_NAMESPACE_END
