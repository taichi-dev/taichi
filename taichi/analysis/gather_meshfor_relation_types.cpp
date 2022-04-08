#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {

class GatherMeshforRelationTypes : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  GatherMeshforRelationTypes() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  static void run(IRNode *root) {
    GatherMeshforRelationTypes analyser;
    root->accept(&analyser);
  }

  void visit(MeshForStmt *stmt) override {
    TI_ASSERT(mesh_for == nullptr);
    TI_ASSERT(stmt->major_to_types.size() == 0);
    TI_ASSERT(stmt->minor_relation_types.size() == 0);
    mesh_for = stmt;
    stmt->body->accept(this);
    mesh_for = nullptr;
  }

  void visit(MeshRelationAccessStmt *stmt) override {
    if (auto from_stmt =
            stmt->mesh_idx->cast<LoopIndexStmt>()) {  // major relation
      TI_ASSERT(from_stmt->mesh_index_type() == mesh_for->major_from_type);
      mesh_for->major_to_types.insert(stmt->to_type);
    } else if (auto from_stmt =
                   stmt->mesh_idx
                       ->cast<MeshRelationAccessStmt>()) {  // minor relation
      TI_ASSERT(!from_stmt->is_size());
      auto from_order = mesh::element_order(from_stmt->to_type);
      auto to_order = mesh::element_order(stmt->to_type);
      TI_ASSERT_INFO(from_order > to_order,
                     "Cannot access an indeterminate relation (E.g, Vert-Vert) "
                     "in a nested neighbor access");
      mesh_for->minor_relation_types.insert(
          mesh::relation_by_orders(from_order, to_order));
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  MeshForStmt *mesh_for{nullptr};
};

void gather_meshfor_relation_types(IRNode *node) {
  GatherMeshforRelationTypes::run(node);
}

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
