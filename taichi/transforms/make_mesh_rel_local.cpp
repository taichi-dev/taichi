#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/transforms/make_mesh_rel_local.h"

namespace taichi {
namespace lang {

namespace {
    // This function creates loop like:
    // int i = start_val;
    // while (i < end_val) {
    //  body(i);
    //  i += blockDim.x;
    // }
    Stmt *create_xlogue(
      OffloadedStmt *offload,
      Stmt *start_val,
      Stmt *end_val,
      std::function<void(Block * /*block*/, Stmt * /*idx_val*/)> body_) {
      Block* block = offload->bls_prologue.get();

      Stmt *idx = block->push_back<AllocaStmt>(PrimitiveType::i32);
      [[maybe_unused]] Stmt *init_val =
          block->push_back<LocalStoreStmt>(idx, start_val);
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
      if (offload->task_type != OffloadedStmt::TaskType::mesh_for)
        return;
      
      if (!offload->major_to_types.size()) { // demoted mesh-for
        return;
      }

      if (offload->major_to_types.size() > 1 || 
          offload->minor_relation_types.size() > 0) {
        TI_NOT_IMPLEMENTED;
      }

      const auto &from_type =  offload->major_from_type;
      const auto &to_type = *offload->major_to_types.begin();
      const auto &from_order = mesh::element_order(from_type);
      const auto &to_order = mesh::element_order(to_type);
      const auto &rel_type = mesh::relation_by_orders(from_order, to_order);
      TI_INFO("rel = <{}, {}>", mesh::element_type_name(from_type), 
                                mesh::element_type_name(to_type));

      if (offload->bls_prologue == nullptr) {
        offload->bls_prologue = std::make_unique<Block>();
        offload->bls_prologue->parent_stmt = offload;
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