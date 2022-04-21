#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/transforms/make_mesh_rel_local.h"

namespace taichi {
namespace lang {

namespace {
    // This function creates loop like:
    // int i = threadIdx.x;
    // while (i < end_val) {
    //  body(i);
    //  i += blockDim.x;
    // }
    Stmt *create_xlogue(
      OffloadedStmt *offload,
      Stmt *end_val,
      std::function<void(Block * /*block*/, Stmt * /*idx_val*/)> body_) {
      Block* block = offload->bls_prologue.get();

      Stmt *idx = block->push_back<AllocaStmt>(PrimitiveType::i32);
      [[maybe_unused]] Stmt *init_val =
          block->push_back<LocalStoreStmt>(idx, 
            block->push_back<LoopLinearIndexStmt>(offload) // Equivalent to CUDA threadIdx
          );
      Stmt *block_dim_val = block->push_back<ConstStmt>(
            LaneAttribute<TypedConstant>{offload->block_dim});

      std::unique_ptr<Block> body = std::make_unique<Block>();
      {
        Stmt *idx_val = body->push_back<LocalLoadStmt>(LocalAddress{idx, 0});
        Stmt *cond =
            body->push_back<BinaryOpStmt>(BinaryOpType::cmp_lt, idx_val, end_val);
        body->push_back<WhileControlStmt>(nullptr, cond);
        body_(body.get(), idx_val);
        Stmt *idx_val_ = body->push_back<BinaryOpStmt>(BinaryOpType::add, idx_val,
                                                      block_dim_val);
        [[maybe_unused]] Stmt *idx_store =
            body->push_back<LocalStoreStmt>(idx, idx_val_);
      }
      block->push_back<WhileStmt>(std::move(body));
      Stmt *idx_val = block->push_back<LocalLoadStmt>(LocalAddress{idx, 0});
      return idx_val;
    }

    void make_mesh_rel_local_offload(OffloadedStmt *offload,
                              const CompileConfig &config,
                              const std::string &kernel_name) {
      if (offload->task_type != OffloadedStmt::TaskType::mesh_for || !offload->major_to_types.size())
        return;

      if (offload->major_to_types.size() > 1 || offload->minor_relation_types.size() > 0) {
        TI_NOT_IMPLEMENTED;
      }

      const auto &from_type =  offload->major_from_type;
      const auto &to_type = *offload->major_to_types.begin();
      const auto &from_order = mesh::element_order(from_type);
      const auto &to_order = mesh::element_order(to_type);
      const auto &rel_type = mesh::relation_by_orders(from_order, to_order);
      TI_INFO("rel = <{}, {}>, max_value_per_patch={}", 
        mesh::element_type_name(from_type), mesh::element_type_name(to_type),
        offload->mesh->relations.find(rel_type)->second.max_value_per_patch);

      if (offload->bls_prologue == nullptr) {
        offload->bls_prologue = std::make_unique<Block>();
        offload->bls_prologue->parent_stmt = offload;
      }

      Block *block = offload->bls_prologue.get();
      
      if (from_order > to_order) {
        Stmt *owned_element_num = offload->owned_num_local.find(from_type)->second;
        size_t to_num = from_type == mesh::MeshElementType::Cell && to_type == mesh::MeshElementType::Edge ? 
                        /*Cell-Edge=*/ 6 : (from_order + 1);
        Stmt *to_num_stmt = block->push_back<ConstStmt>(LaneAttribute<TypedConstant>{to_num});
        Stmt *patch_rel_num = block->push_back<BinaryOpStmt>(BinaryOpType::mul, owned_element_num, to_num_stmt);
        Stmt *total_element_offset = offload->total_offset_local.find(from_type)->second;

        // Allocate bls
        DataType dt = PrimitiveType::u16;
        auto dsize = data_type_size(dt);
        // Ensure BLS alignment
        offload->bls_size += (dsize - offload->bls_size % dsize) % dsize;
        const auto & rel_offset_in_bytes = offload->bls_size;
        // Allocate storage for the BLS variable
        offload->bls_size += dsize * offload->mesh->relations.find(rel_type)->second.max_value_per_patch;
        TI_INFO("bls size = {}", offload->bls_size);

        Stmt *offset =
          block->push_back<ConstStmt>(TypedConstant(int32(rel_offset_in_bytes)));
        
        SNode *rel_value = offload->mesh->relations.find(rel_type)->second.value;

        // Fetch relation data to shared mem
        create_xlogue(
          offload,
          patch_rel_num,
          [&](Block *body, Stmt *idx_val) {
            // Global ptr
            Stmt *global_offset = body->push_back<BinaryOpStmt>(BinaryOpType::add, idx_val, 
              body->push_back<BinaryOpStmt>(BinaryOpType::mul, to_num_stmt, total_element_offset));
            Stmt *global_ptr = body->push_back<GlobalPtrStmt>(
            LaneAttribute<SNode *>{rel_value}, std::vector<Stmt *>{global_offset});
            // Global value
            Stmt *value = body->push_back<GlobalLoadStmt>(global_ptr);

            // SM ptr
            Stmt *idx_val_byte = body->push_back<BinaryOpStmt>(BinaryOpType::mul, idx_val,
              body->push_back<ConstStmt>(TypedConstant(dsize)));
            Stmt *index = body->push_back<BinaryOpStmt>(BinaryOpType::add, offset, idx_val_byte);
            Stmt *bls_ptr = body->push_back<BlockLocalPtrStmt>(index, TypeFactory::create_vector_or_scalar_type(1, dt, true));
            body->push_back<GlobalStoreStmt>(bls_ptr, value);
          }
        );
      }
  }
}

const PassID MakeMeshRelLocalPass::id = "MakeMeshRelLocalPass";

namespace irpass {

void make_mesh_rel_local(IRNode *root,
                      const CompileConfig &config,
                      const MakeMeshRelLocalPass::Args &args) {
  TI_AUTO_PROF;

  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      make_mesh_rel_local_offload(offload->cast<OffloadedStmt>(), config,
                               args.kernel_name);
    }
  } else {
    make_mesh_rel_local_offload(root->as<OffloadedStmt>(), config,
                             args.kernel_name);
  }
  type_check(root, config);
}

}

}
}