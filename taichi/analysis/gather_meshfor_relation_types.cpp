#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

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

    // Check metadata available
    std::set<mesh::MeshElementType> all_elements;
    all_elements.insert(mesh_for->major_from_type);
    for (auto _type : mesh_for->major_to_types) {
      all_elements.insert(_type);
    }
    for (auto _type : all_elements) {
      TI_ERROR_IF(mesh_for->mesh->num_elements.find(_type) ==
                      mesh_for->mesh->num_elements.end(),
                  "Cannot load mesh element {}'s metadata",
                  mesh::element_type_name(_type));
    }

    std::set<mesh::MeshRelationType> all_relations;
    for (auto _type : mesh_for->major_to_types) {
      all_relations.insert(
          mesh::relation_by_orders(int(mesh_for->major_from_type), int(_type)));
    }
    for (auto _type : mesh_for->minor_relation_types) {
      all_relations.insert(_type);
    }

    bool missing = false;
    std::string full_name;
    std::string short_name;
    for (auto _type : all_relations) {
      if (mesh_for->mesh->relations.find(_type) ==
          mesh_for->mesh->relations.end()) {
        if (missing) {
          full_name += ", ";
          short_name += ", ";
        }
        full_name += mesh::relation_type_name(_type);
        short_name += '\'';
        short_name += char(mesh::element_type_name(mesh::MeshElementType(
                               mesh::from_end_element_order(_type)))[0] +
                           'A' - 'a');
        short_name += char(mesh::element_type_name(mesh::MeshElementType(
                               mesh::to_end_element_order(_type)))[0] +
                           'A' - 'a');
        short_name += '\'';
        missing = true;
      }
    }

    if (missing) {
      TI_ERROR(
          "Relation {} detected in mesh-for loop but not initialized."
          " Please add them with syntax: Patcher.load_mesh(..., "
          "relations=[..., {}])",
          full_name, short_name);
    }

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

}  // namespace taichi::lang
