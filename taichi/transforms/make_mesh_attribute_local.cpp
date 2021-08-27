#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/scratch_pad.h"
#include "taichi/transforms/make_mesh_attribute_local.h"

namespace taichi {
namespace lang {

const PassID MakeMeshAttributeLocal::id = "MakeMeshAttributeLocal";

namespace irpass {

void make_mesh_attribute_local_offload(OffloadedStmt *offload,
                              const CompileConfig &config,
                              const std::string &kernel_name) {
  if (offload->task_type != OffloadedStmt::TaskType::mesh_for) {
    return;
  }

  for (int j = 0; j < (int)offload->body->statements.size(); j++) {
    if (auto ifs = offload->body->statements[j]->cast<InternalFuncStmt>()) {
      if (ifs->func_name == "l2g") {
        VecStatement block;
        SNode *l2g = ifs->args[0]->cast<GlobalLoadStmt>()->src->cast<GlobalPtrStmt>()->snodes[0];
        SNode *offset = ifs->args[2]->cast<GlobalLoadStmt>()->src->cast<GlobalPtrStmt>()->snodes[0];
        auto get_load = [&](SNode *snode, Stmt* idx) {
          const auto lane = std::vector<Stmt*>{idx};
          Stmt* globalptr = block.push_back<GlobalPtrStmt>(LaneAttribute<SNode*>{snode}, lane);
          Stmt* load = block.push_back<GlobalLoadStmt>(globalptr);
          return load;
        };
        Stmt* mesh_idx = block.push_back<InternalFuncStmt, const std::string, const std::vector<Stmt*>>("mesh_idx", {});
        Stmt* total_v = get_load(offset, mesh_idx);
        Stmt* index = block.push_back<BinaryOpStmt>(BinaryOpType::add, total_v, ifs->args[1]);
        Stmt* ans = get_load(l2g, index);
        ifs->replace_with(std::move(block));
      }
    }
  }
  offload->bls_prologue = std::make_unique<Block>();
  offload->bls_prologue->parent_stmt = offload;
  Stmt* mesh_idx = offload->bls_prologue->push_back<InternalFuncStmt, const std::string, const std::vector<Stmt*>>("mesh_idx", {});
  Stmt* one = offload->bls_prologue->push_back<ConstStmt>(LaneAttribute<TypedConstant>(
    TypedConstant(TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32), 1)));
  Stmt* mesh_idx_1 = offload->bls_prologue->push_back<BinaryOpStmt>(BinaryOpType::add, mesh_idx, one);
  auto get_print = [&](Stmt* idx) {
    const auto lane = std::vector<Stmt*>{idx};
    Stmt* globalptr = offload->bls_prologue->push_back<GlobalPtrStmt>(LaneAttribute<SNode*>{offload->snode}, lane);
    Stmt* load = offload->bls_prologue->push_back<GlobalLoadStmt>(globalptr);
    offload->bls_prologue->push_back<PrintStmt>(load);
  };
  get_print(mesh_idx);
  get_print(mesh_idx_1);
}

// This pass should happen after offloading but before lower_access
void make_mesh_attribute_local(IRNode *root,
                      const CompileConfig &config,
                      const MakeBlockLocalPass::Args &args) {
  TI_AUTO_PROF;

  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      make_mesh_attribute_local_offload(offload->cast<OffloadedStmt>(), config,
                               args.kernel_name);
    }
  } else {
    make_mesh_attribute_local_offload(root->as<OffloadedStmt>(), config,
                             args.kernel_name);
  }
  type_check(root, config);
}

}  // namespace irpass

}
}
