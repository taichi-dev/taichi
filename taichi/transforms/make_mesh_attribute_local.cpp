#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/transforms/make_mesh_attribute_local.h"
#include "taichi/ir/visitors.h"

namespace taichi {
namespace lang {

const PassID MakeMeshAttributeLocal::id = "MakeMeshAttributeLocal";

namespace irpass {

class ReplaceRelationAccess : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ReplaceRelationAccess(OffloadedStmt *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    offload = node;
    visit(node);
  }

  OffloadedStmt *offload;

  void visit(MeshRelationAccessStmt *stmt) override {
    mesh::MeshElementType from_type;
    if (auto idx = stmt->mesh_idx->cast<LoopIndexStmt>()) {
      from_type = idx->mesh_index_type();
    } else if (auto idx = stmt->mesh_idx->cast<MeshRelationAccessStmt>()) {
      from_type = idx->to_type;
    } else {
      TI_NOT_IMPLEMENTED;
    }

    auto from_order = mesh::element_order(from_type);
    auto to_order = mesh::element_order(stmt->to_type);
    mesh::MeshRelationType rel_type =
        mesh::relation_by_orders(from_order, to_order);
    if (from_order > to_order) {  // high-to-low relation
      SNode *rel_value = stmt->mesh->relations.find(rel_type)->second.value;
      VecStatement block;
      auto get_load = [&](SNode *snode, Stmt *idx) {
        const auto lane = std::vector<Stmt *>{idx};
        Stmt *globalptr =
            block.push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{snode}, lane);
        Stmt *load = block.push_back<GlobalLoadStmt>(globalptr);
        return load;
      };
      Stmt *to_size = block.push_back<ConstStmt>(LaneAttribute<TypedConstant>{
          from_type == mesh::MeshElementType::Cell &&
                  stmt->to_type == mesh::MeshElementType::Edge
              ? /*Cell-Edge=*/6
              : (from_order + 1)});
      // E.g, v_2 = CV[(c + owned_cells_offset) * 4 + 2]
      Stmt *offset = stmt->mesh->owned_offset_local.find(from_type)->second;
      Stmt *tmp0 = block.push_back<BinaryOpStmt>(BinaryOpType::add, offset,
                                                 stmt->mesh_idx);
      Stmt *tmp1 =
          block.push_back<BinaryOpStmt>(BinaryOpType::mul, tmp0, to_size);
      Stmt *index = block.push_back<BinaryOpStmt>(BinaryOpType::add, tmp1,
                                                  stmt->neighbor_idx);
      Stmt *val = get_load(rel_value, index);
      stmt->replace_with(std::move(block));
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }
};

class ReplaceIndexConversion : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ReplaceIndexConversion(OffloadedStmt *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    offload = node;
    visit(node);
  }

  void visit(MeshIndexConversionStmt *stmt) override {
    SNode *mapping = nullptr;
    mesh::MeshElementType element_type;

    if (auto idx = stmt->idx->cast<LoopIndexStmt>()) {
      element_type = idx->mesh_index_type();
    } else if (auto idx = stmt->idx->cast<MeshRelationAccessStmt>()) {
      element_type = idx->to_type;
    } else {
      TI_NOT_IMPLEMENTED;
    }

    if (stmt->conv_type == mesh::ConvType::l2g) {
      mapping = stmt->mesh->l2g_map.find(element_type)->second;
    } else if (stmt->conv_type == mesh::ConvType::l2r) {
      mapping = stmt->mesh->l2r_map.find(element_type)->second;
    } else {
      TI_NOT_IMPLEMENTED;
    }

    VecStatement block;
    auto get_load = [&](SNode *snode, Stmt *idx) {
      const auto lane = std::vector<Stmt *>{idx};
      Stmt *globalptr =
          block.push_back<GlobalPtrStmt>(LaneAttribute<SNode *>{snode}, lane);
      Stmt *load = block.push_back<GlobalLoadStmt>(globalptr);
      return load;
    };
    // E.g, v_global = v_l2g[v_local + total_vertices_offset]
    Stmt *offset = stmt->mesh->total_offset_local.find(element_type)->second;
    Stmt *index =
        block.push_back<BinaryOpStmt>(BinaryOpType::add, stmt->idx, offset);
    Stmt *val = get_load(mapping, index);
    stmt->replace_with(std::move(block));
  }

  OffloadedStmt *offload;
};

void make_mesh_attribute_local_offload(OffloadedStmt *offload,
                                       const CompileConfig &config,
                                       const std::string &kernel_name) {
  if (offload->task_type != OffloadedStmt::TaskType::mesh_for) {
    return;
  }

  ReplaceIndexConversion rep1(offload);
  ReplaceRelationAccess rep2(offload);
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
}  // namespace lang
}  // namespace taichi
