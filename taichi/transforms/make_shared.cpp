#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

namespace {

void make_shared_offload(OffloadedStmt *offload) {
  if (offload->task_type != offload->struct_for)
    return;

  std::size_t shared_offset = 0;

  /*
  // reduce buffer fragmentation.
  for (auto dest : valid_reduction_values) {
    auto data_type = dest->ret_type.data_type;
    auto dtype_size = data_type_size(data_type);
    // Step 1:
    // Fetch to shared memory storage
    {
      if (offload->prologue == nullptr) {
        offload->prologue = std::make_unique<Block>();
      }

      // ensure alignment
      // tls_offset += (dtype_size - tls_offset % dtype_size) % dtype_size;



      auto tls_ptr = offload->prologue->push_back<ThreadLocalPtrStmt>(
          tls_offset, VectorType(1, data_type));

      auto zero = offload->prologue->insert(
          std::make_unique<ConstStmt>(TypedConstant(data_type, 0)), -1);
      // Zero-fill
      // TODO: do not use GlobalStore for TLS ptr.
      offload->prologue->push_back<GlobalStoreStmt>(tls_ptr, zero);
    }

    // Step 2:
    // Make loop body accumulate to TLS ptr instead of global ptr
    {
      auto tls_ptr = offload->body->insert(
          Stmt::make<ThreadLocalPtrStmt>(tls_offset, VectorType(1, data_type)),
          0);
      dest->replace_with(tls_ptr);
    }

    // Step 3:
    // Atomic-add thread local contribution to its global version
    {
      if (offload->epilogue == nullptr) {
        offload->epilogue = std::make_unique<Block>();
      }
      auto tls_ptr = offload->epilogue->push_back<ThreadLocalPtrStmt>(
          tls_offset, VectorType(1, data_type));
      // TODO: do not use global load from TLS.
      auto tls_load = offload->epilogue->push_back<GlobalLoadStmt>(tls_ptr);
      auto global_ptr = offload->epilogue->insert(
          std::unique_ptr<Stmt>(
              (Stmt *)irpass::analysis::clone(dest).release()),
          -1);
      offload->epilogue->push_back<AtomicOpStmt>(AtomicOpType::add, global_ptr,
                                                 tls_load);
    }

    // allocate storage for the TLS variable
    tls_offset += dtype_size;
  }

  offload->tls_size = std::max(std::size_t(1), tls_offset);
  */
}

}  // namespace

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_shared(IRNode *root) {
  TI_AUTO_PROF;
  auto root_block = root->cast<Block>();
  TI_ASSERT(root_block);
  for (auto &offload : root_block->statements) {
    make_shared_offload(offload->cast<OffloadedStmt>());
  }
  typecheck(root);
  fix_block_parents(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
