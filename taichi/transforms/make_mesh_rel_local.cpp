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
    void create_xlogue(
      OffloadedStmt *offload,
      Stmt *end_val,
      std::function<void(Block * /*block*/, Stmt * /*idx_val*/)> body_) {
      Block* block = offload->bls_prologue.get();

      Stmt *idx = block->push_back<AllocaStmt>(PrimitiveType::i32);
      block->push_back<LocalStoreStmt>(idx, block->push_back<LoopLinearIndexStmt>(offload) /*Equivalent to CUDA threadIdx*/);
      Stmt *block_dim_val = block->push_back<ConstStmt>(LaneAttribute<TypedConstant>{offload->block_dim});

      std::unique_ptr<Block> body = std::make_unique<Block>();
      {
        Stmt *idx_val = body->push_back<LocalLoadStmt>(LocalAddress{idx, 0});
        Stmt *cond = body->push_back<BinaryOpStmt>(BinaryOpType::cmp_lt, idx_val, end_val);
        body->push_back<WhileControlStmt>(nullptr, cond);
        body_(body.get(), idx_val);
        Stmt *idx_val_ = body->push_back<BinaryOpStmt>(BinaryOpType::add, idx_val, block_dim_val);
        [[maybe_unused]] Stmt *idx_store = body->push_back<LocalStoreStmt>(idx, idx_val_);
      }
      block->push_back<WhileStmt>(std::move(body));
    }

    void make_mesh_rel_local_offload(OffloadedStmt *offload,
                              const CompileConfig &config,
                              const std::string &kernel_name) {
      if (offload->task_type != OffloadedStmt::TaskType::mesh_for || !offload->major_to_types.size())
        return;

      if (offload->major_to_types.size() > 1 || offload->minor_relation_types.size() > 0) {
        return;
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
        int to_num = from_type == mesh::MeshElementType::Cell && to_type == mesh::MeshElementType::Edge ? 
                        /*Cell-Edge=*/ 6 : (from_order + 1);
        Stmt *to_num_stmt = block->push_back<ConstStmt>(LaneAttribute<TypedConstant>{int32(to_num)});
        Stmt *patch_rel_num = block->push_back<BinaryOpStmt>(BinaryOpType::mul, owned_element_num, to_num_stmt);    
        Stmt *total_element_offset = offload->total_offset_local.find(from_type)->second;
        Stmt *inital_offset = block->push_back<BinaryOpStmt>(BinaryOpType::mul, to_num_stmt, total_element_offset);

        // Allocate bls
        DataType dt = PrimitiveType::u16;
        auto dsize = data_type_size(dt);
        // Ensure BLS alignment
        offload->bls_size += (dsize - offload->bls_size % dsize) % dsize;
        const auto rel_offset_in_bytes = offload->bls_size;
        TI_INFO("rel_offset_in_bytes = {}", rel_offset_in_bytes);
        // Allocate storage for the BLS variable
        offload->bls_size += dsize * offload->mesh->relations.find(rel_type)->second.max_value_per_patch;
        TI_INFO("bls size = {}", offload->bls_size);

        Stmt *offset = block->push_back<ConstStmt>(TypedConstant(int32(rel_offset_in_bytes)));
        SNode *rel_value = offload->mesh->relations.find(rel_type)->second.value;

        // Fetch relation data to shared mem
        create_xlogue(
          offload,
          patch_rel_num,
          [&](Block *body, Stmt *idx_val) {
            // Global ptr
            Stmt *global_offset = body->push_back<BinaryOpStmt>(BinaryOpType::add, idx_val, inital_offset);
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

            /*
            Stmt *cond1 = body->push_back<BinaryOpStmt>(BinaryOpType::cmp_eq, inital_offset, body->push_back<ConstStmt>(LaneAttribute<TypedConstant>{928924}));
            Stmt *ifst = body->push_back<IfStmt>(cond1);
            std::unique_ptr<Block> if_body = std::make_unique<Block>();
            if_body->push_back<PrintStmt>("global_idx = ", global_offset, ", local_idx = ", index, ", value = ", value, "\n");
            ifst->cast<IfStmt>()->set_true_statements(std::move(if_body));
            */
          }
        );

        // Replace relation access statement
        std::vector<MeshRelationAccessStmt *> rel_access_stmts;
        irpass::analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
          if (auto rel_access = stmt->cast<MeshRelationAccessStmt>()) {
              rel_access_stmts.push_back(rel_access);
          }
          return false;
        });

        for (auto stmt : rel_access_stmts) {
          VecStatement bls;
          Stmt *to_size = bls.push_back<ConstStmt>(LaneAttribute<TypedConstant>{int32(to_num)});

          
          Stmt *bls_offset_bytes = bls.push_back<ConstStmt>(LaneAttribute<TypedConstant>{int32(rel_offset_in_bytes)});
          Stmt *idx = bls.push_back<BinaryOpStmt>(BinaryOpType::add,
            bls.push_back<BinaryOpStmt>(BinaryOpType::mul, stmt->mesh_idx, bls.push_back<ConstStmt>(LaneAttribute<TypedConstant>{int32(to_num)})),
            stmt->neighbor_idx);
          Stmt *idx_bytes = bls.push_back<BinaryOpStmt>(BinaryOpType::mul, idx, bls.push_back<ConstStmt>(TypedConstant(dsize)));
          Stmt *final_offset = bls.push_back<BinaryOpStmt>(BinaryOpType::add, bls_offset_bytes, idx_bytes);
          
          Stmt *bls_ptr = bls.push_back<BlockLocalPtrStmt>(final_offset, TypeFactory::create_vector_or_scalar_type(1, dt, true));
          Stmt *value = bls.push_back<GlobalLoadStmt>(bls_ptr);

          /*
          // E.g, v_2 = CV[(c + total_cells_offset) * 4 + 2]
          Stmt *tmp0 = bls.push_back<BinaryOpStmt>(BinaryOpType::add, total_element_offset, stmt->mesh_idx);
          Stmt *tmp1 = bls.push_back<BinaryOpStmt>(BinaryOpType::mul, tmp0, to_size);
          Stmt *index = bls.push_back<BinaryOpStmt>(BinaryOpType::add, tmp1, stmt->neighbor_idx);
          Stmt *global_ptr = bls.push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{rel_value}, std::vector<Stmt *>{index});
          Stmt *value_global = bls.push_back<GlobalLoadStmt>(global_ptr);

          Stmt *cond = bls.push_back<BinaryOpStmt>(BinaryOpType::cmp_eq, value, value_global);
          Stmt *cond1 = bls.push_back<BinaryOpStmt>(BinaryOpType::cmp_eq, inital_offset, bls.push_back<ConstStmt>(LaneAttribute<TypedConstant>{928924}));
          IfStmt *ifst = bls.push_back<IfStmt>(cond1);
          std::unique_ptr<Block> if_body = std::make_unique<Block>();
          Stmt *assert_st = if_body->push_back<AssertStmt>(cond, "Fuck error=%d,l_idx=%d right=%d,g_idx=%d, total_offset=%d, mesh_idx=%d, neighor_idx=%d, patch_rel_num=%d, own_element_num=%d, initial_offset=%d", 
            std::vector<Stmt *>({value, final_offset, value_global, index, total_element_offset, stmt->mesh_idx, stmt->neighbor_idx, patch_rel_num, owned_element_num, inital_offset}));
          ifst->set_true_statements(std::move(if_body));
          Stmt *value1 = bls.push_back<BinaryOpStmt>(BinaryOpType::add, value_global, bls.push_back<ConstStmt>(LaneAttribute<TypedConstant>{0}));
          */
  
          stmt->replace_with(std::move(bls));
        }
      } else {
        Stmt *owned_num = offload->owned_num_local.find(from_type)->second;
        Stmt *owned_offset = offload->owned_offset_local.find(from_type)->second;
        Stmt *_1 = block->push_back<ConstStmt>(TypedConstant(1));
        Stmt *owned_num_1 = block->push_back<BinaryOpStmt>(BinaryOpType::add, owned_num, _1);
        Stmt *patch_idx = block->push_back<MeshPatchIndexStmt>();
        Stmt *owned_offset_patch_idx = block->push_back<BinaryOpStmt>(BinaryOpType::add, owned_offset, patch_idx);

        SNode *rel_offset = offload->mesh->relations.find(rel_type)->second.offset;
        SNode *rel_value = offload->mesh->relations.find(rel_type)->second.value;

        // Patch_offset, load once
        Stmt *patch_offset = block->push_back<GlobalLoadStmt>(
                        block->push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{offload->mesh->relations.find(rel_type)->second.patch_offset}, std::vector<Stmt *>{patch_idx}));
        
        DataType dt = PrimitiveType::u16;
        auto dsize = data_type_size(dt);
        // Allocate bls for `offset`
        // Ensure BLS alignment
        offload->bls_size += (dsize - offload->bls_size % dsize) % dsize;
        const auto offset_offset_in_bytes = offload->bls_size;
        TI_INFO("rel(offset)_offset_in_bytes = {}", offset_offset_in_bytes);
        // Allocate storage for the `offset` BLS variable
        offload->bls_size += dsize * (offload->mesh->patch_max_element_num.find(from_type)->second + 1);
        TI_INFO("bls size = {}", offload->bls_size);

        // Allocate bls for `value`
        // Ensure BLS alignment
        offload->bls_size += (dsize - offload->bls_size % dsize) % dsize;
        const auto value_offset_in_bytes = offload->bls_size;
        TI_INFO("rel(value)_offset_in_bytes = {}", value_offset_in_bytes);
        // Allocate storage for the `offset` BLS variable
        offload->bls_size += dsize * offload->mesh->relations.find(rel_type)->second.max_value_per_patch;
        TI_INFO("bls size = {}", offload->bls_size);

        Stmt *offset_offset_bytes = block->push_back<ConstStmt>(TypedConstant(int32(offset_offset_in_bytes)));
        Stmt *value_offset_bytes = block->push_back<ConstStmt>(TypedConstant(int32(value_offset_in_bytes)));
        Stmt *dsize_value = block->push_back<ConstStmt>(TypedConstant(dsize));

        // Fetch relation `offset` data to shared mem
        create_xlogue(
          offload,
          owned_num_1,
          [&](Block *body, Stmt *idx_val) {
            // Global ptr
            Stmt *global_offset = body->push_back<BinaryOpStmt>(BinaryOpType::add, idx_val, owned_offset_patch_idx);
            Stmt *global_ptr = body->push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{rel_offset}, std::vector<Stmt *>{global_offset});
            // Global value
            Stmt *value = body->push_back<GlobalLoadStmt>(global_ptr);

            // SM ptr
            Stmt *idx_val_byte = body->push_back<BinaryOpStmt>(BinaryOpType::mul, idx_val, dsize_value);
            Stmt *index = body->push_back<BinaryOpStmt>(BinaryOpType::add, offset_offset_bytes, idx_val_byte);
            Stmt *bls_ptr = body->push_back<BlockLocalPtrStmt>(index, TypeFactory::create_vector_or_scalar_type(1, dt, true));
            body->push_back<GlobalStoreStmt>(bls_ptr, value);
          }
        );

        Stmt *owned_offset_patch_idx_own_element = block->push_back<BinaryOpStmt>(BinaryOpType::add, owned_offset_patch_idx, owned_num);
        Stmt *offset_end = block->push_back<GlobalLoadStmt>(
                        block->push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{offload->mesh->relations.find(rel_type)->second.offset}, std::vector<Stmt *>{owned_offset_patch_idx_own_element}));
        create_xlogue(
          offload,
          offset_end,
          [&](Block *body, Stmt *idx_val) {
            // Global ptr
            Stmt *global_offset = body->push_back<BinaryOpStmt>(BinaryOpType::add, idx_val, patch_offset);
            Stmt *global_ptr = body->push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{rel_value}, std::vector<Stmt *>{global_offset});
            // Global value
            Stmt *value = body->push_back<GlobalLoadStmt>(global_ptr);

            // SM ptr
            Stmt *idx_val_byte = body->push_back<BinaryOpStmt>(BinaryOpType::mul, idx_val, dsize_value);
            Stmt *index = body->push_back<BinaryOpStmt>(BinaryOpType::add, value_offset_bytes, idx_val_byte);
            Stmt *bls_ptr = body->push_back<BlockLocalPtrStmt>(index, TypeFactory::create_vector_or_scalar_type(1, dt, true));
            body->push_back<GlobalStoreStmt>(bls_ptr, value);
          }
        );

        // Replace relation access statement
        std::vector<MeshRelationAccessStmt *> rel_access_stmts;
        irpass::analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
          if (auto rel_access = stmt->cast<MeshRelationAccessStmt>()) {
              rel_access_stmts.push_back(rel_access);
          }
          return false;
        });

        for (auto stmt : rel_access_stmts) {
          VecStatement bls;
          if (stmt->is_size()) {
            Stmt *idx_0 = bls.push_back<BinaryOpStmt>(BinaryOpType::add, bls.push_back<BinaryOpStmt>(BinaryOpType::mul, stmt->mesh_idx, dsize_value), offset_offset_bytes);
            Stmt *idx_1 = bls.push_back<BinaryOpStmt>(BinaryOpType::add, idx_0, dsize_value);
            Stmt *bls_value_0 = bls.push_back<GlobalLoadStmt>(bls.push_back<BlockLocalPtrStmt>(idx_0, TypeFactory::create_vector_or_scalar_type(1, dt, true)));
            Stmt *bls_value_1 = bls.push_back<GlobalLoadStmt>(bls.push_back<BlockLocalPtrStmt>(idx_1, TypeFactory::create_vector_or_scalar_type(1, dt, true)));
            Stmt *value = bls.push_back<BinaryOpStmt>(BinaryOpType::sub, bls_value_1, bls_value_0);
          } else {
            Stmt *idx_0 = bls.push_back<BinaryOpStmt>(BinaryOpType::add, bls.push_back<BinaryOpStmt>(BinaryOpType::mul, stmt->mesh_idx, dsize_value), offset_offset_bytes);
            Stmt *offset = bls.push_back<GlobalLoadStmt>(bls.push_back<BlockLocalPtrStmt>(idx_0, TypeFactory::create_vector_or_scalar_type(1, dt, true)));
            Stmt *offset_neighbor = bls.push_back<BinaryOpStmt>(BinaryOpType::add, offset, stmt->neighbor_idx);
            Stmt *offset_neighbor_offset = bls.push_back<BinaryOpStmt>(BinaryOpType::add, bls.push_back<BinaryOpStmt>(BinaryOpType::mul, offset_neighbor, dsize_value), value_offset_bytes);
            Stmt *value = bls.push_back<GlobalLoadStmt>(bls.push_back<BlockLocalPtrStmt>(offset_neighbor_offset, TypeFactory::create_vector_or_scalar_type(1, dt, true)));
          }
          stmt->replace_with(std::move(bls));
        }
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