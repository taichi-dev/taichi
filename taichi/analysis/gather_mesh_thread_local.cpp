#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

using MeshElementTypeSet = std::unordered_set<mesh::MeshElementType>;

class GatherMeshThreadLocal : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  GatherMeshThreadLocal(OffloadedStmt *offload_,
                        MeshElementTypeSet *owned_ptr_,
                        MeshElementTypeSet *total_ptr_) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    this->offload = offload_;
    this->owned_ptr = owned_ptr_;
    this->total_ptr = total_ptr_;
  }

  static void run(OffloadedStmt *offload,
                  MeshElementTypeSet *owned_ptr,
                  MeshElementTypeSet *total_ptr) {
    TI_ASSERT(offload->task_type == OffloadedStmt::TaskType::mesh_for);
    GatherMeshThreadLocal analyser(offload, owned_ptr, total_ptr);
    offload->accept(&analyser);
  }

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->is_mesh_index()) {
      this->owned_ptr->insert(stmt->mesh_index_type());
    }
  }

  void visit(MeshRelationAccessStmt *stmt) override {
    if (auto idx = stmt->mesh_idx->cast<LoopIndexStmt>()) {
      this->owned_ptr->insert(idx->mesh_index_type());
    } else if (auto idx = stmt->mesh_idx->cast<MeshRelationAccessStmt>()) {
      this->owned_ptr->insert(idx->to_type);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(MeshRelationSizeStmt *stmt) override {
    if (auto idx = stmt->mesh_idx->cast<LoopIndexStmt>()) {
      this->owned_ptr->insert(idx->mesh_index_type());
    } else if (auto idx = stmt->mesh_idx->cast<MeshRelationAccessStmt>()) {
      this->owned_ptr->insert(idx->to_type);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(MeshIndexConversionStmt *stmt) override {
    if (auto idx = stmt->idx->cast<LoopIndexStmt>()) {
      this->total_ptr->insert(idx->mesh_index_type());
    } else if (auto idx = stmt->idx->cast<MeshRelationAccessStmt>()) {
      this->total_ptr->insert(idx->to_type);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  OffloadedStmt *offload;
  MeshElementTypeSet *owned_ptr;
  MeshElementTypeSet *total_ptr;
};

namespace irpass::analysis {

std::pair</* owned= */ MeshElementTypeSet,
          /* total= */ MeshElementTypeSet>
gather_mesh_thread_local(OffloadedStmt *offload) {
  MeshElementTypeSet local_owned{};
  MeshElementTypeSet local_total{};

  GatherMeshThreadLocal::run(offload, &local_owned, &local_total);
  return std::make_pair(local_owned, local_total);
}

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
