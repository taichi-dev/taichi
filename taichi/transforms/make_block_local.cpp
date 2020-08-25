#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/scratch_pad.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

namespace {

void make_block_local_offload(OffloadedStmt *offload) {
  if (offload->task_type != OffloadedStmt::TaskType::struct_for)
    return;

  bool debug = offload->get_kernel()->program.config.debug;

  auto pads = irpass::initialize_scratch_pad(offload);

  std::size_t bls_offset = 0;

  for (auto &pad : pads->pads) {
    auto snode = pad.first;
    auto data_type = snode->dt;
    auto dtype_size = data_type_size(data_type);

    bool bls_has_read = pad.second.total_flags & AccessFlag::read;
    bool bls_has_write = pad.second.total_flags & AccessFlag::write;
    bool bls_has_accumulate = pad.second.total_flags & AccessFlag::accumulate;

    TI_ASSERT_INFO(!bls_has_write, "BLS with write accesses is not supported.")
    TI_ASSERT_INFO(!(bls_has_accumulate && bls_has_read),
                   "BLS with both read and accumulation is not supported.")

    // dim = Dimensionality of the BLS buffer and the block
    auto dim = (int)pad.second.pad_size.size();
    TI_ASSERT(dim == snode->num_active_indices);

    auto bls_num_elements = pad.second.pad_size_linear();

    std::vector<int> block_strides(dim);
    std::vector<int> bls_strides(dim);
    block_strides[dim - 1] = 1;
    bls_strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--) {
      // TODO: fix the virtual/physical index correspondence here
      // TODO: rename "pad"
      // "pad" is the BLS buffer ("scratch pad")
      block_strides[i] = block_strides[i + 1] * pad.second.block_size[i + 1];
      bls_strides[i] = bls_strides[i + 1] * pad.second.pad_size[i + 1];
    }

    // TODO: improve IR builder to make this part easier to read

    // Ensure BLS alignment
    bls_offset += (dtype_size - bls_offset % dtype_size) % dtype_size;

    // This lambda is used for both BLS prologue and epilogue creation
    auto create_xlogue =
        [&](std::unique_ptr<Block> &block,
            const std::function<void(
                Block * element_block, std::vector<Stmt *> global_indices,
                Stmt * bls_element_offset_bytes)> &operation) {
          if (block == nullptr) {
            block = std::make_unique<Block>();
          }
          Stmt *block_linear_index =
              block->push_back<LoopLinearIndexStmt>(offload);

          /*
          Note that since there are fewer elements in the block than in BLS,
          each thread may have to fetch more than one element to BLS.
          Therefore on CUDA we need something like

          auto bls_element_id = block_linear_index;
          while (bls_element_id < bls_size) {
            i, j, k = bls_to_global(bls_element_id)
            bls[bls_element_id] = x[i, j, k]
            // or x[i, j, k] = bls[bls_element_id]
            bls_element_id += block_dim;
          }

          Since we know block_dim and bls_size at compile time and there's
          usually not too many iterations, we directly unroll this while loop
          for performance when constructing prologues/epilogues.
          */

          // Unroll the while-loop
          int loop_offset = 0;
          int block_dim = offload->block_dim;
          while (loop_offset < bls_num_elements) {
            Block *element_block = nullptr;
            auto loop_offset_stmt =
                block->push_back<ConstStmt>(TypedConstant(loop_offset));

            auto bls_element_id_this_iteration = block->push_back<BinaryOpStmt>(
                BinaryOpType::add, loop_offset_stmt, block_linear_index);

            auto bls_element_offset_bytes = block->push_back<BinaryOpStmt>(
                BinaryOpType::mul, bls_element_id_this_iteration,
                block->push_back<ConstStmt>(TypedConstant(dtype_size)));

            bls_element_offset_bytes = block->push_back<BinaryOpStmt>(
                BinaryOpType::add, bls_element_offset_bytes,
                block->push_back<ConstStmt>(TypedConstant((int32)bls_offset)));

            if (loop_offset + block_dim > bls_num_elements) {
              // Need to create an IfStmt to safeguard since bls size may not be
              // a multiple of block_size, and this iteration some threads may
              // go over bls_num_elements ("block-stride" loop)
              auto cond = block->push_back<BinaryOpStmt>(
                  BinaryOpType::cmp_lt, bls_element_id_this_iteration,
                  block->push_back<ConstStmt>(TypedConstant(bls_num_elements)));
              auto if_stmt =
                  dynamic_cast<IfStmt *>(block->push_back<IfStmt>(cond));
              if_stmt->true_statements = std::make_unique<Block>();
              element_block = if_stmt->true_statements.get();
            } else {
              // No need to create an if since every thread is within
              // bls_num_elements.
              element_block = block.get();
            }

            std::vector<Stmt *> global_indices(dim);

            // Convert bls_element_id to global indices
            // via a series of % and /.
            auto bls_element_id_partial = bls_element_id_this_iteration;
            for (int i = dim - 1; i >= 0; i--) {
              auto size = element_block->push_back<ConstStmt>(
                  TypedConstant(pad.second.pad_size[i]));

              auto bls_coord = element_block->push_back<BinaryOpStmt>(
                  BinaryOpType::mod, bls_element_id_partial, size);
              bls_element_id_partial = element_block->push_back<BinaryOpStmt>(
                  BinaryOpType::div, bls_element_id_partial, size);

              auto global_index = element_block->push_back<BinaryOpStmt>(
                  BinaryOpType::add,
                  element_block->push_back<ConstStmt>(
                      TypedConstant(pad.second.bounds[0][i])),
                  bls_coord);

              global_index = element_block->push_back<BinaryOpStmt>(
                  BinaryOpType::add, global_index,
                  element_block->push_back<BlockCornerIndexStmt>(offload, i));

              global_indices[i] = global_index;
            }

            operation(element_block, global_indices, bls_element_offset_bytes);
            // TODO: do not use GlobalStore for BLS ptr.

            loop_offset += block_dim;
          }
        };

    // Step 1:
    // Fetch to BLS
    {
      create_xlogue(
          offload->bls_prologue,
          [&](Block *element_block, std::vector<Stmt *> global_indices,
              Stmt *bls_element_offset_bytes) {
            Stmt *value;
            if (bls_has_read) {
              // Read access
              // Fetch from global to BLS

              auto global_pointer = element_block->push_back<GlobalPtrStmt>(
                  snode, global_indices);
              value = element_block->push_back<GlobalLoadStmt>(global_pointer);
            } else {
              // Accumulation access
              // Zero-fill
              value = element_block->push_back<ConstStmt>(
                  TypedConstant(data_type, 0));
            }
            auto bls_ptr = element_block->push_back<BlockLocalPtrStmt>(
                bls_element_offset_bytes, VectorType(1, data_type));
            element_block->push_back<GlobalStoreStmt>(bls_ptr, value);
          });
    }

    // Step 2:
    // Make loop body load from BLS instead of global fields
    {
      std::vector<GlobalPtrStmt *> global_ptrs;

      // TODO: no more abuse of gather_statements...
      irpass::analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
        if (auto global_ptr = stmt->cast<GlobalPtrStmt>()) {
          TI_ASSERT(global_ptr->width() == 1);
          if (global_ptr->snodes[0] == snode) {
            global_ptrs.push_back(global_ptr);
          }
        }
        return false;
      });

      for (auto global_ptr : global_ptrs) {
        VecStatement bls;
        Stmt *bls_element_offset = nullptr;
        auto global_indices = global_ptr->indices;
        for (int i = 0; i < dim; i++) {
          // BLS index = sum_i inc_i
          // where inc_i =
          //   bls_stride_i * (gbl_idx_i - loop_base_i - bls_lower_bound_i)
          // Note that when index offsets are used, the offset contributions are
          // already included in bls_lower_bound_i.
          auto block_corner = bls.push_back<BlockCornerIndexStmt>(offload, i);

          auto inc = bls.push_back<BinaryOpStmt>(
              BinaryOpType::sub, global_indices[i], block_corner);
          inc = bls.push_back<BinaryOpStmt>(
              BinaryOpType::sub, inc,
              bls.push_back<ConstStmt>(TypedConstant(pad.second.bounds[0][i])));

          if (debug) {
            // This part insert an assertion to make sure BLS access is within
            // the bound.
            auto bls_axis_size =
                pad.second.bounds[1][i] - pad.second.bounds[0][i];
            std::string msg = fmt::format(
                "(kernel={}, body) Access out of bound: BLS buffer axis {} "
                "(size {}) with "
                "index %d.",
                offload->get_kernel()->name, i, bls_axis_size);

            auto lower_bound =
                bls.push_back<ConstStmt>(LaneAttribute<TypedConstant>(0));
            auto check_lower_bound = bls.push_back<BinaryOpStmt>(
                BinaryOpType::cmp_ge, inc, lower_bound);

            auto upper_bound = bls.push_back<ConstStmt>(
                LaneAttribute<TypedConstant>(bls_axis_size));
            auto check_upper_bound = bls.push_back<BinaryOpStmt>(
                BinaryOpType::cmp_lt, inc, upper_bound);

            auto check_i = bls.push_back<BinaryOpStmt>(
                BinaryOpType::bit_and, check_lower_bound, check_upper_bound);

            bls.push_back<AssertStmt>(check_i, msg, std::vector<Stmt *>{inc});
          }

          inc = bls.push_back<BinaryOpStmt>(
              BinaryOpType::mul, inc,
              bls.push_back<ConstStmt>(TypedConstant(bls_strides[i])));

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
        global_ptr->replace_with(std::move(bls));
      }
    }

    // Step 3:
    // Atomic-add BLS contribution to its global version if necessary
    if (bls_has_accumulate) {
      create_xlogue(
          offload->bls_epilogue,
          [&](Block *element_block, std::vector<Stmt *> global_indices,
              Stmt *bls_element_offset_bytes) {
            // Store/accumulate from BLS to global
            auto bls_ptr = element_block->push_back<BlockLocalPtrStmt>(
                bls_element_offset_bytes, VectorType(1, data_type));
            auto bls_val = element_block->push_back<GlobalLoadStmt>(bls_ptr);

            auto global_pointer =
                element_block->push_back<GlobalPtrStmt>(snode, global_indices);
            element_block->push_back<AtomicOpStmt>(AtomicOpType::add,
                                                   global_pointer, bls_val);
          });
    }

    // allocate storage for the BLS variable
    bls_offset += dtype_size * bls_num_elements;
  }

  offload->bls_size = std::max(std::size_t(1), bls_offset);
}

}  // namespace

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_block_local(IRNode *root) {
  TI_AUTO_PROF;

  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      make_block_local_offload(offload->cast<OffloadedStmt>());
    }
  } else {
    make_block_local_offload(root->as<OffloadedStmt>());
  }
  type_check(root);
  fix_block_parents(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
