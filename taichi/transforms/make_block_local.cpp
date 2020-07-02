#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/scratch_pad.h"

TLANG_NAMESPACE_BEGIN

namespace {

void make_block_local_offload(OffloadedStmt *offload) {
  if (offload->task_type != offload->struct_for)
    return;

  /*
  for (auto s : offload->scratch_opt) {
    if (s.first != 0) {  // 0 means shared
      continue;
    }
    TI_INFO("Emitting shared memory for {}",
            s.second->get_node_type_name_hinted());

    auto snode = s.second;
  }
  */
  auto pads = irpass::initialize_scratch_pad(offload);

  std::size_t bls_offset = 0;

  for (auto &pad : pads->pads) {
    auto snode = pad.first;
    auto data_type = snode->dt;
    auto dtype_size = data_type_size(data_type);

    auto get_stride = [&](int i) {
      // TODO: fix the index correspondence here
      int stride = 1;
      for (int j = i + 1; j < pad.second.pad_size.size(); j++) {
        stride *= pad.second.block_size[j];
      }
      return stride;
    };

    // Step 1:
    // Fetch to block local memory storage
    {
      if (offload->prologue == nullptr) {
        offload->prologue = std::make_unique<Block>();
      }
      auto block = offload->prologue.get();

      // ensure alignment
      bls_offset += (dtype_size - bls_offset % dtype_size) % dtype_size;

      Stmt *linear_index = nullptr;

      for (int i = 0; i < pad.second.pad_size.size(); i++) {
        // TODO: fix the index correspondence here
        auto inc = block->push_back<BinaryOpStmt>(
            BinaryOpType::mul,
            block->push_back<ConstStmt>(TypedConstant(get_stride(i))),
            block->push_back<BinaryOpStmt>(
                BinaryOpType::sub, block->push_back<LoopIndexStmt>(offload, i),
                block->push_back<LoopIndexBaseStmt>(offload, i)));
        if (linear_index) {
          linear_index = block->push_back<BinaryOpStmt>(BinaryOpType::add,
                                                        linear_index, inc);
        } else {
          linear_index = inc;
        }
      }

      // Unroll the loading while-loop here
      int loop_offset = 0;
      while (loop_offset < pad.second.pad_size_linear()) {
        Block *element_block = nullptr;
        auto loop_offset_stmt =
            block->push_back<ConstStmt>(TypedConstant(loop_offset));
        auto scratch_element_id = linear_index;
        if (loop_offset + pad.second.block_size_linear() >
            pad.second.pad_size_linear()) {
          // Need to create an IfStmt to safeguard
          auto cond = block->push_back<BinaryOpStmt>(
              BinaryOpType::cmp_lt, scratch_element_id,
              block->push_back<ConstStmt>(
                  TypedConstant(pad.second.pad_size_linear())));
          auto if_stmt = dynamic_cast<IfStmt *>(block->push_back<IfStmt>(cond));
          if_stmt->true_statements = std::make_unique<Block>();
          element_block = if_stmt->true_statements.get();
        } else {
          element_block = block;
        }

        auto bls_index = element_block->push_back<BinaryOpStmt>(
            BinaryOpType::add, loop_offset_stmt, linear_index);
        auto bls_index_bytes = element_block->push_back<BinaryOpStmt>(
            BinaryOpType::mul, bls_index,
            element_block->push_back<ConstStmt>(TypedConstant(dtype_size)));

        std::vector<Stmt *> global_indices;
        auto partial_indices = scratch_element_id;
        for (int i = 0; i < pad.second.pad_size.size(); i++) {
          auto size = element_block->push_back<ConstStmt>(
              TypedConstant(pad.second.pad_size[i]));
          auto scratch_index = element_block->push_back<BinaryOpStmt>(
              BinaryOpType::mod, partial_indices, size);
          auto global_index = element_block->push_back<BinaryOpStmt>(
              BinaryOpType::add,
              element_block->push_back<ConstStmt>(
                  TypedConstant(pad.second.bounds[0][i])),
              scratch_index);
          global_index = element_block->push_back<BinaryOpStmt>(
              BinaryOpType::add, global_index,
              element_block->push_back<LoopIndexBaseStmt>(offload, i));
          global_indices.push_back(global_index);
        }
        // Recompute global indices
        // TODO: do not use GlobalStore for BLS ptr.
        auto glb_ptr =
            element_block->push_back<GlobalPtrStmt>(snode, global_indices);
        auto load = element_block->push_back<GlobalLoadStmt>(glb_ptr);
        auto bls_ptr = element_block->push_back<BlockLocalPtrStmt>(
            bls_index_bytes, VectorType(1, data_type));
        element_block->push_back<GlobalStoreStmt>(bls_ptr, load);
        loop_offset += pad.second.block_size_linear();
        scratch_element_id = block->push_back<BinaryOpStmt>(
            BinaryOpType::add, scratch_element_id,
            block->push_back<ConstStmt>(
                TypedConstant(pad.second.block_size_linear())));
      }
    }

    // Step 2:
    // Make loop body load BLS ptr instead of global ptr
    {
      std::vector<GlobalPtrStmt *> global_ptrs;
      // TODO: no abuse of gather_statements...
      irpass::analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
        if (auto global_ptr = stmt->cast<GlobalPtrStmt>()) {
          TI_ASSERT(global_ptr->width() == 1);
          if (global_ptr->snodes[0] == snode) {
            global_ptrs.push_back(global_ptr);
          }
        }
        return false;
      });

      for (auto glb_ptr : global_ptrs) {
        VecStatement bls;
        Stmt *bls_element_offset = nullptr;
        for (int i = 0; i < pad.second.pad_size.size(); i++) {
          auto global_indices = glb_ptr->indices;
          auto inc = bls.push_back<BinaryOpStmt>(
              BinaryOpType::sub, global_indices[i],
              bls.push_back<LoopIndexBaseStmt>(offload, i));
          inc = bls.push_back<BinaryOpStmt>(
              BinaryOpType::mul, inc,
              bls.push_back<ConstStmt>(TypedConstant(get_stride(i))));
          if (!bls_element_offset) {
            bls_element_offset = inc;
          } else {
            bls_element_offset = bls.push_back<BinaryOpStmt>(
                BinaryOpType::add, bls_element_offset, inc);
          }
        }

        // convert to bytes
        bls_element_offset = bls.push_back<BinaryOpStmt>(
            BinaryOpType::mul, bls_element_offset,
            bls.push_back<ConstStmt>(TypedConstant(dtype_size)));

        // add array offset
        bls_element_offset = bls.push_back<BinaryOpStmt>(
            BinaryOpType::add, bls_element_offset,
            bls.push_back<ConstStmt>(TypedConstant((int32)bls_offset)));

        bls.push_back<BlockLocalPtrStmt>(bls_element_offset,
                                         VectorType(1, data_type));
        glb_ptr->replace_with(std::move(bls));
      }
    }

    // Step 3:
    // Atomic-add block local contribution to its global version
    if (pad.second.total_flags & AccessFlag::write) {
      TI_NOT_IMPLEMENTED
      /*
      //
      if (offload->epilogue == nullptr) {
        offload->epilogue = std::make_unique<Block>();
      }
      auto bls_ptr = offload->epilogue->push_back<BlockLocalPtrStmt>(
          bls_offset, VectorType(1, data_type));
      // TODO: do not use global load from BLS.
      auto tls_load = offload->epilogue->push_back<GlobalLoadStmt>(bls_ptr);
      auto global_ptr = offload->epilogue->insert(
          std::unique_ptr<Stmt>(
              (Stmt *)irpass::analysis::clone(dest).release()),
          -1);
      offload->epilogue->push_back<AtomicOpStmt>(AtomicOpType::add, global_ptr,
                                                 tls_load);
                                                 */
    }

    // allocate storage for the BLS variable
    bls_offset += dtype_size * pad.second.pad_size_linear();
  }

  offload->bls_size = std::max(std::size_t(1), bls_offset);
}

}  // namespace

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_block_local(IRNode *root) {
  TI_AUTO_PROF;
  auto root_block = root->cast<Block>();
  TI_ASSERT(root_block);
  for (auto &offload : root_block->statements) {
    make_block_local_offload(offload->cast<OffloadedStmt>());
  }
  typecheck(root);
  fix_block_parents(root);
  irpass::re_id(root);
  irpass::print(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
