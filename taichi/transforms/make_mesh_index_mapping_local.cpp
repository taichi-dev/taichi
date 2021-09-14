#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/transforms/make_mesh_index_mapping_local.h"

namespace taichi {
namespace lang {

const PassID MakeMeshIndexMappingLocal::id = "MakeMeshIndexMappingLocal";

void MakeMeshIndexMappingLocal::simplify_nested_conversion() {
  std::vector<MeshIndexConversionStmt *> stmts;
  std::vector<Stmt *> ori_indices;

  irpass::analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
    if (auto conv1 = stmt->cast<MeshIndexConversionStmt>()) {
      if (auto conv2 = conv1->idx->cast<MeshIndexConversionStmt>()) {
        if (conv1->conv_type == mesh::ConvType::g2r &&
            conv2->conv_type == mesh::ConvType::l2g &&
            conv1->mesh == conv2->mesh &&
            conv1->idx_type == conv2->idx_type) {  // nested
          stmts.push_back(conv1);
          ori_indices.push_back(conv2->idx);
        }
      }
    }
    return false;
  });

  for (size_t i = 0; i < stmts.size(); ++i) {
    stmts[i]->replace_with(Stmt::make<MeshIndexConversionStmt>(
        stmts[i]->mesh, stmts[i]->idx_type, ori_indices[i],
        mesh::ConvType::l2r));
  }
}

void MakeMeshIndexMappingLocal::fetch_mapping_to_bls(
    mesh::MeshElementType element_type,
    mesh::ConvType conv_type) {
  auto &block = offload->bls_prologue;
  auto bls_mapping_loop =
      [&](Stmt *start_val, Stmt *end_val,
          std::function<Stmt *(Block *, Stmt *)> global_val) {
        Stmt *idx = block->push_back<AllocaStmt>(data_type);
        [[maybe_unused]] Stmt *init_val =
            block->push_back<LocalStoreStmt>(idx, start_val);
        Stmt *bls_element_offset_bytes = block->push_back<ConstStmt>(
            LaneAttribute<TypedConstant>{(int32)bls_offset_in_bytes});
        Stmt *block_dim_val = block->push_back<ConstStmt>(
            LaneAttribute<TypedConstant>{offload->block_dim});

        std::unique_ptr<Block> body = std::make_unique<Block>();
        {
          Stmt *idx_val = body->push_back<LocalLoadStmt>(LocalAddress{idx, 0});
          Stmt *cond = body->push_back<BinaryOpStmt>(BinaryOpType::cmp_lt,
                                                     idx_val, end_val);
          { body->push_back<WhileControlStmt>(nullptr, cond); }
          Stmt *idx_val_byte = body->push_back<BinaryOpStmt>(
              BinaryOpType::mul, idx_val,
              body->push_back<ConstStmt>(TypedConstant(dtype_size)));
          Stmt *offset = body->push_back<BinaryOpStmt>(
              BinaryOpType::add, bls_element_offset_bytes, idx_val_byte);
          Stmt *bls_ptr = body->push_back<BlockLocalPtrStmt>(
              offset,
              TypeFactory::create_vector_or_scalar_type(1, data_type, true));
          [[maybe_unused]] Stmt *bls_store = body->push_back<GlobalStoreStmt>(
              bls_ptr, global_val(body.get(), idx_val));

          Stmt *idx_val_ = body->push_back<BinaryOpStmt>(
              BinaryOpType::add, idx_val, block_dim_val);
          [[maybe_unused]] Stmt *idx_store =
              body->push_back<LocalStoreStmt>(idx, idx_val_);
        }
        block->push_back<WhileStmt>(std::move(body));
        Stmt *idx_val = block->push_back<LocalLoadStmt>(LocalAddress{idx, 0});
        return idx_val;
      };

  Stmt *thread_idx_stmt = block->push_back<LoopLinearIndexStmt>(
      offload);  // Equivalent to CUDA threadIdx
  Stmt *total_element_num = offload->total_num_local.find(element_type)->second;
  Stmt *total_element_offset =
      offload->total_offset_local.find(element_type)->second;

  if (config.optimize_mesh_reordered_mapping &&
      conv_type == mesh::ConvType::l2r) {
    // int i = threadIdx.x;
    // while (i < owned_{}_num) {
    //  mapping_shared[i] = i + owned_{}_offset;
    //  i += blockDim.x;
    // }
    // while (i < total_{}_num) {
    //  mapping_shared[i] = mapping[i + total_{}_offset];
    //  i += blockDim.x;
    // }
    Stmt *owned_element_num =
        offload->owned_num_local.find(element_type)->second;
    Stmt *owned_element_offset =
        offload->owned_offset_local.find(element_type)->second;
    Stmt *pre_idx_val = bls_mapping_loop(
        thread_idx_stmt, owned_element_num, [&](Block *body, Stmt *idx_val) {
          Stmt *global_index = body->push_back<BinaryOpStmt>(
              BinaryOpType::add, idx_val, owned_element_offset);
          return global_index;
        });
    bls_mapping_loop(
        pre_idx_val, total_element_num, [&](Block *body, Stmt *idx_val) {
          Stmt *global_offset = body->push_back<BinaryOpStmt>(
              BinaryOpType::add, total_element_offset, idx_val);
          Stmt *global_ptr = body->push_back<GlobalPtrStmt>(
              LaneAttribute<SNode *>{snode},
              std::vector<Stmt *>{global_offset});
          Stmt *global_load = body->push_back<GlobalLoadStmt>(global_ptr);
          return global_load;
        });
  } else {
    // int i = threadIdx.x;
    // while (i < total_{}_num) {
    //  mapping_shared[i] = mapping[i + total_{}_offset];
    //  i += blockDim.x;
    // }
    bls_mapping_loop(
        thread_idx_stmt, total_element_num, [&](Block *body, Stmt *idx_val) {
          Stmt *global_offset = body->push_back<BinaryOpStmt>(
              BinaryOpType::add, total_element_offset, idx_val);
          Stmt *global_ptr = body->push_back<GlobalPtrStmt>(
              LaneAttribute<SNode *>{snode},
              std::vector<Stmt *>{global_offset});
          Stmt *global_load = body->push_back<GlobalLoadStmt>(global_ptr);
          return global_load;
        });
  }
}

void MakeMeshIndexMappingLocal::replace_conv_statements(
    mesh::MeshElementType element_type,
    mesh::ConvType conv_type) {
  std::vector<MeshIndexConversionStmt *> idx_conv_stmts;

  irpass::analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
    if (auto idx_conv = stmt->cast<MeshIndexConversionStmt>()) {
      if (idx_conv->mesh == offload->mesh && idx_conv->conv_type == conv_type &&
          idx_conv->idx_type == element_type) {
        idx_conv_stmts.push_back(idx_conv);
      }
    }
    return false;
  });

  for (auto stmt : idx_conv_stmts) {
    VecStatement bls;
    Stmt *bls_element_offset_bytes = bls.push_back<ConstStmt>(
        LaneAttribute<TypedConstant>{(int32)bls_offset_in_bytes});
    Stmt *idx_byte = bls.push_back<BinaryOpStmt>(
        BinaryOpType::mul, stmt->idx,
        bls.push_back<ConstStmt>(TypedConstant(dtype_size)));
    Stmt *offset = bls.push_back<BinaryOpStmt>(
        BinaryOpType::add, bls_element_offset_bytes, idx_byte);
    Stmt *bls_ptr = bls.push_back<BlockLocalPtrStmt>(
        offset, TypeFactory::create_vector_or_scalar_type(1, data_type, true));
    [[maybe_unused]] Stmt *bls_load = bls.push_back<GlobalLoadStmt>(bls_ptr);
    stmt->replace_with(std::move(bls));
  }
}

MakeMeshIndexMappingLocal::MakeMeshIndexMappingLocal(
    OffloadedStmt *offload,
    const CompileConfig &config)
    : offload(offload), config(config) {
  simplify_nested_conversion();

  // TODO(changyu): A analyzer to determinte which mapping should be localized
  mappings.insert(std::make_pair(mesh::MeshElementType::Vertex,
                                 mesh::ConvType::l2g));  // FIXME: A hack

  bls_offset_in_bytes = offload->bls_size;
  auto &block = offload->bls_prologue;

  for (auto [element_type, conv_type] : mappings) {
    // There is not corresponding mesh element attribute read/write,
    // It's useless to localize this mapping
    if (offload->total_offset_local.find(element_type) ==
        offload->total_offset_local.end()) {
      continue;
    }

    TI_ASSERT(conv_type != mesh::ConvType::g2r);  // g2r will not be cached.
    snode = (offload->mesh->index_mapping
                 .find(std::make_pair(element_type, conv_type))
                 ->second);
    data_type = snode->dt.ptr_removed();
    dtype_size = data_type_size(data_type);

    // Ensure BLS alignment
    bls_offset_in_bytes +=
        (dtype_size - bls_offset_in_bytes % dtype_size) % dtype_size;

    if (block == nullptr) {
      block = std::make_unique<Block>();
      block->parent_stmt = offload;
    }

    // Step 1:
    // Fetch index mapping to the BLS block first
    fetch_mapping_to_bls(element_type, conv_type);

    // Step 2:
    // Make mesh index mapping load from BLS instead of global fields

    // TODO(changyu): before this step, if a mesh attribute field needs to be
    // localized, We should simply remove the `MeshIndexConversionStmt`
    replace_conv_statements(element_type, conv_type);

    // allocate storage for the BLS variable
    bls_offset_in_bytes +=
        dtype_size *
        offload->mesh->patch_max_element_num.find(element_type)->second;
  }

  offload->bls_size = std::max(std::size_t(1), bls_offset_in_bytes);
}

void MakeMeshIndexMappingLocal::run(OffloadedStmt *offload,
                                    const CompileConfig &config,
                                    const std::string &kernel_name) {
  if (offload->task_type != OffloadedStmt::TaskType::mesh_for) {
    return;
  }

  MakeMeshIndexMappingLocal instance(offload, config);
}

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_mesh_index_mapping_local(
    IRNode *root,
    const CompileConfig &config,
    const MakeMeshIndexMappingLocal::Args &args) {
  TI_AUTO_PROF;

  // =========================================================================================
  // This pass generates code like this:
  // // Load V_l2g
  // for (int i = threadIdx.x; i < total_vertices; i += blockDim.x) {
  //   V_l2g[i] = _V_l2g[i + total_vertices_offset];
  // }
  // =========================================================================================

  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      MakeMeshIndexMappingLocal::run(offload->cast<OffloadedStmt>(), config,
                                     args.kernel_name);
    }
  } else {
    MakeMeshIndexMappingLocal::run(root->as<OffloadedStmt>(), config,
                                   args.kernel_name);
  }

  type_check(root, config);
}

}  // namespace irpass
}  // namespace lang
}  // namespace taichi
