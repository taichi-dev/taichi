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
                        MeshElementTypeSet *total_ptr_,
                        bool optimize_mesh_reordered_mapping_) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    this->offload = offload_;
    this->owned_ptr = owned_ptr_;
    this->total_ptr = total_ptr_;
    this->optimize_mesh_reordered_mapping = optimize_mesh_reordered_mapping_;
  }

  static void run(OffloadedStmt *offload,
                  MeshElementTypeSet *owned_ptr,
                  MeshElementTypeSet *total_ptr,
                  const CompileConfig &config) {
    TI_ASSERT(offload->task_type == OffloadedStmt::TaskType::mesh_for);
    GatherMeshThreadLocal analyser(offload, owned_ptr, total_ptr,
                                   config.optimize_mesh_reordered_mapping);
    offload->accept(&analyser);
  }

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->is_mesh_index()) {
      this->owned_ptr->insert(stmt->mesh_index_type());
    }
  }

  void visit(MeshRelationAccessStmt *stmt) override {
    if (mesh::element_order(stmt->from_type()) >
        mesh::element_order(stmt->to_type)) {
      this->total_ptr->insert(stmt->from_type());
    } else {
      this->owned_ptr->insert(stmt->from_type());
    }
  }

  void visit(MeshIndexConversionStmt *stmt) override {
    this->total_ptr->insert(stmt->idx_type);
    if (optimize_mesh_reordered_mapping &&
        stmt->conv_type == mesh::ConvType::l2r) {
      this->owned_ptr->insert(stmt->idx_type);
    }
  }

  OffloadedStmt *offload{nullptr};
  MeshElementTypeSet *owned_ptr{nullptr};
  MeshElementTypeSet *total_ptr{nullptr};
  bool optimize_mesh_reordered_mapping{false};
};

namespace irpass::analysis {

std::pair</* owned= */ MeshElementTypeSet,
          /* total= */ MeshElementTypeSet>
gather_mesh_thread_local(OffloadedStmt *offload, const CompileConfig &config) {
  MeshElementTypeSet local_owned{};
  MeshElementTypeSet local_total{};

  GatherMeshThreadLocal::run(offload, &local_owned, &local_total, config);
  return std::make_pair(local_owned, local_total);
}

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
