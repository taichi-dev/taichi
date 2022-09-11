#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/transforms/demote_mesh_statements.h"
#include "taichi/ir/visitors.h"

namespace taichi {
namespace lang {

const PassID DemoteMeshStatements::id = "DemoteMeshStatements";

namespace irpass {

auto get_load = [](SNode *snode, Stmt *idx, VecStatement &block) {
  const auto lane = std::vector<Stmt *>{idx};
  Stmt *globalptr = block.push_back<GlobalPtrStmt>(snode, lane);
  Stmt *load = block.push_back<GlobalLoadStmt>(globalptr);
  return load;
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
    SNode *mapping = (stmt->mesh->index_mapping
                          .find(std::make_pair(stmt->idx_type, stmt->conv_type))
                          ->second);

    VecStatement block;
    if (stmt->conv_type == mesh::ConvType::g2r) {
      // E.g, v_reordered = v_g2r[v_global]
      [[maybe_unused]] Stmt *val = get_load(mapping, stmt->idx, block);
    } else {
      // E.g, v_global = v_l2g[v_local + total_vertices_offset]
      Stmt *offset = offload->total_offset_local.find(stmt->idx_type)->second;
      Stmt *index =
          block.push_back<BinaryOpStmt>(BinaryOpType::add, stmt->idx, offset);
      [[maybe_unused]] Stmt *val = get_load(mapping, index, block);
    }
    stmt->replace_with(std::move(block));
  }

  OffloadedStmt *offload;
};

void demote_mesh_statements_offload(OffloadedStmt *offload,
                                    const CompileConfig &config,
                                    const std::string &kernel_name) {
  ReplaceIndexConversion rep_conv(
      offload);  // This demote should work for any offloaed statement

  if (offload->task_type != OffloadedStmt::TaskType::mesh_for) {
    return;
  }

  auto stmts = irpass::analysis::gather_statements(
      offload->body.get(),
      [&](Stmt *stmt) { return stmt->is<MeshRelationAccessStmt>(); });

  for (int i = stmts.size() - 1; i >= 0; --i) {
    auto stmt = stmts[i]->cast<MeshRelationAccessStmt>();
    mesh::MeshElementType from_type = stmt->from_type();

    auto from_order = mesh::element_order(from_type);
    auto to_order = mesh::element_order(stmt->to_type);
    mesh::MeshRelationType rel_type =
        mesh::relation_by_orders(from_order, to_order);
    if (from_order > to_order) {  // high-to-low relation
      if (stmt->is_size()) {
        stmt->replace_with(Stmt::make<ConstStmt>(
            TypedConstant{from_type == mesh::MeshElementType::Cell &&
                                  stmt->to_type == mesh::MeshElementType::Edge
                              ? /*Cell-Edge=*/6
                              : (from_order + 1)}));
      } else {
        SNode *rel_value = stmt->mesh->relations.find(rel_type)->second.value;
        VecStatement block;
        Stmt *to_size = block.push_back<ConstStmt>(
            TypedConstant{from_type == mesh::MeshElementType::Cell &&
                                  stmt->to_type == mesh::MeshElementType::Edge
                              ? /*Cell-Edge=*/6
                              : (from_order + 1)});
        // E.g, v_2 = CV[(c + total_cells_offset) * 4 + 2]
        Stmt *offset = offload->total_offset_local.find(from_type)->second;
        Stmt *tmp0 = block.push_back<BinaryOpStmt>(BinaryOpType::add, offset,
                                                   stmt->mesh_idx);
        Stmt *tmp1 =
            block.push_back<BinaryOpStmt>(BinaryOpType::mul, tmp0, to_size);
        Stmt *index = block.push_back<BinaryOpStmt>(BinaryOpType::add, tmp1,
                                                    stmt->neighbor_idx);
        [[maybe_unused]] Stmt *val = get_load(rel_value, index, block);
        stmt->replace_with(std::move(block));
      }
    } else {  // low-to-high or same-order
      const auto &rel = stmt->mesh->relations.find(rel_type)->second;
      SNode *rel_offset = rel.offset;
      SNode *rel_patch_offset = rel.patch_offset;
      VecStatement block;
      Stmt *patch_idx = block.push_back<MeshPatchIndexStmt>();
      Stmt *owned_offset = offload->owned_offset_local.find(from_type)->second;
      Stmt *patch_offset = get_load(rel_patch_offset, patch_idx, block);
      Stmt *index_offset = block.push_back<BinaryOpStmt>(
          BinaryOpType::add, patch_idx, owned_offset);
      Stmt *index = block.push_back<BinaryOpStmt>(BinaryOpType::add,
                                                  index_offset, stmt->mesh_idx);
      Stmt *offset = get_load(rel_offset, index, block);
      if (stmt->is_size()) {
        Stmt *one = block.push_back<ConstStmt>(TypedConstant{1});
        Stmt *index_1 =
            block.push_back<BinaryOpStmt>(BinaryOpType::add, index, one);
        Stmt *offset_1 = get_load(rel_offset, index_1, block);
        [[maybe_unused]] Stmt *val =
            block.push_back<BinaryOpStmt>(BinaryOpType::sub, offset_1, offset);
      } else {
        SNode *rel_value = stmt->mesh->relations.find(rel_type)->second.value;
        Stmt *val_local_index = block.push_back<BinaryOpStmt>(
            BinaryOpType::add, offset, stmt->neighbor_idx);
        Stmt *val_index = block.push_back<BinaryOpStmt>(
            BinaryOpType::add, val_local_index, patch_offset);
        [[maybe_unused]] Stmt *val = get_load(rel_value, val_index, block);
      }
      stmt->replace_with(std::move(block));
    }
  }
}

// This pass should happen after offloading but before lower_access
void demote_mesh_statements(IRNode *root,
                            const CompileConfig &config,
                            const DemoteMeshStatements::Args &args) {
  TI_AUTO_PROF;

  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      demote_mesh_statements_offload(offload->cast<OffloadedStmt>(), config,
                                     args.kernel_name);
    }
  } else {
    demote_mesh_statements_offload(root->as<OffloadedStmt>(), config,
                                   args.kernel_name);
  }
  type_check(root, config);
}

}  // namespace irpass
}  // namespace lang
}  // namespace taichi
