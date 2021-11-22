#include "taichi/analysis/mesh_bls_analyzer.h"

#include "taichi/system/profiler.h"
#include "taichi/ir/analysis.h"

namespace taichi {
namespace lang {

MeshBLSAnalyzer::MeshBLSAnalyzer(OffloadedStmt *for_stmt, MeshBLSCaches *caches)
    : for_stmt_(for_stmt), caches_(caches) {
  TI_AUTO_PROF;
  allow_undefined_visitor = true;
  invoke_default_visitor = false;
}

void MeshBLSAnalyzer::record_access(Stmt *stmt, AccessFlag flag) {
  if (!analysis_ok_) {
    return;
  }
  if (!stmt->is<GlobalPtrStmt>())
    return;  // local alloca
  auto ptr = stmt->as<GlobalPtrStmt>();
  if (ptr->indices.size() != std::size_t(1) ||
      !ptr->indices[0]->is<MeshIndexConversionStmt>())
    return;
  auto conv = ptr->indices[0]->as<MeshIndexConversionStmt>();
  auto element_type = conv->idx_type;
  auto conv_type = conv->conv_type;
  if (conv_type == mesh::ConvType::g2r)
    return;
  for (int l = 0; l < stmt->width(); l++) {
    auto snode = ptr->snodes[l];
    if (!caches_->has(snode)) {
      continue;
    }

    if (!caches_->access(snode, element_type, conv_type, flag)) {
      analysis_ok_ = false;
      break;
    }
  }
}

void MeshBLSAnalyzer::visit(GlobalLoadStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);  // TODO: support vectorization
  record_access(stmt->src, AccessFlag::read);
}

void MeshBLSAnalyzer::visit(GlobalStoreStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);  // TODO: support vectorization
  record_access(stmt->dest, AccessFlag::write);
}

void MeshBLSAnalyzer::visit(AtomicOpStmt *stmt) {
  if (stmt->op_type == AtomicOpType::add) {
    record_access(stmt->dest, AccessFlag::accumulate);
  }
}

void MeshBLSAnalyzer::visit(Stmt *stmt) {
  TI_ASSERT(!stmt->is_container_statement());
}

bool MeshBLSAnalyzer::run() {
  const auto &block = for_stmt_->body;

  for (int i = 0; i < (int)block->statements.size(); i++) {
    block->statements[i]->accept(this);
  }

  return analysis_ok_;
}

namespace irpass {
namespace analysis {

std::unique_ptr<MeshBLSCaches> initialize_mesh_local_attribute(
    OffloadedStmt *offload) {
  TI_AUTO_PROF
  TI_ASSERT(offload->task_type == OffloadedTaskType::mesh_for);
  std::unique_ptr<MeshBLSCaches> caches;
  caches = std::make_unique<MeshBLSCaches>();
  for (auto snode : offload->mem_access_opt.get_snodes_with_flag(
           SNodeAccessFlag::mesh_local)) {
    caches->insert(snode);
  }

  MeshBLSAnalyzer bls_analyzer(offload, caches.get());
  bool analysis_ok = bls_analyzer.run();
  if (!analysis_ok) {
    TI_ERROR("Mesh BLS analysis failed !");
  }
  return caches;
}

}  // namespace analysis
}  // namespace irpass

}  // namespace lang
}  // namespace taichi
