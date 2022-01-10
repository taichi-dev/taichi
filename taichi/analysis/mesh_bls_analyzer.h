#pragma once

#include "taichi/program/compile_config.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/mesh.h"

#include <set>

namespace taichi {
namespace lang {

class MeshBLSCache {
 public:
  using AccessFlag = taichi::lang::AccessFlag;
  using Rec = std::map<std::pair<mesh::MeshElementType, mesh::ConvType>,
                       std::set<std::pair<SNode *, AccessFlag>>>;

  SNode *snode{nullptr};
  mesh::MeshElementType element_type;
  mesh::ConvType conv_type;

  bool initialized;
  bool finalized;
  bool loop_index;
  int unique_accessed;
  AccessFlag total_flags;

  MeshBLSCache() = default;

  MeshBLSCache(SNode *snode) : snode(snode) {
    total_flags = AccessFlag(0);
    initialized = false;
    finalized = false;
    loop_index = false;
    unique_accessed = 0;
  }

  bool access(mesh::MeshElementType element_type,
              mesh::ConvType conv_type,
              AccessFlag flags,
              Stmt *idx) {
    if (!initialized) {
      initialized = true;
      this->conv_type = conv_type;
      this->element_type = element_type;
    } else {
      if (this->conv_type != conv_type || this->element_type != element_type)
        return false;
    }
    this->total_flags |= flags;
    if (idx->is<LoopIndexStmt>()) {
      loop_index = true;
    } else {
      unique_accessed++;
    }
    return true;
  }

  void finalize(Rec &rec) {
    TI_ASSERT(!finalized);
    finalized = true;
    if (initialized) {
      const auto cache_type = std::make_pair(element_type, conv_type);
      auto ptr = rec.find(cache_type);
      if (ptr == rec.end()) {
        ptr = rec.emplace(std::piecewise_construct,
                          std::forward_as_tuple(cache_type),
                          std::forward_as_tuple())
                  .first;
      }
      ptr->second.insert(std::make_pair(snode, total_flags));
    }
  }
};

class MeshBLSCaches {
 public:
  std::map<SNode *, MeshBLSCache> caches;

  using AccessFlag = MeshBLSCache::AccessFlag;
  using Rec = MeshBLSCache::Rec;

  void insert(SNode *snode) {
    if (caches.find(snode) == caches.end()) {
      caches.emplace(std::piecewise_construct, std::forward_as_tuple(snode),
                     std::forward_as_tuple(snode));
    } else {
      TI_ERROR("mesh::MeshBLSCaches for {} already exists.",
               snode->node_type_name);
    }
  }

  bool access(SNode *snode,
              mesh::MeshElementType element_type,
              mesh::ConvType conv_type,
              AccessFlag flags,
              Stmt *idx) {
    if (caches.find(snode) == caches.end())
      return false;
    return caches.find(snode)->second.access(element_type, conv_type, flags,
                                             idx);
  }

  Rec finalize() {
    Rec rec;
    for (auto &cache : caches) {
      cache.second.finalize(rec);
    }
    return rec;
  }

  bool has(SNode *snode) {
    return caches.find(snode) != caches.end();
  }

  MeshBLSCache &get(SNode *snode) {
    TI_ASSERT(caches.find(snode) != caches.end());
    return caches[snode];
  }
};

// Figure out accessed SNodes, and their ranges in this for stmt
class MeshBLSAnalyzer : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 public:
  MeshBLSAnalyzer(OffloadedStmt *for_stmt,
                  MeshBLSCaches *caches,
                  bool auto_mesh_local,
                  const CompileConfig &config);

  void visit(GlobalPtrStmt *stmt) override {
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override;

  void visit(GlobalStoreStmt *stmt) override;

  void visit(AtomicOpStmt *stmt) override;

  void visit(Stmt *stmt) override;

  bool run();

 private:
  void record_access(Stmt *stmt, AccessFlag flag);

  OffloadedStmt *for_stmt_{nullptr};
  MeshBLSCaches *caches_{nullptr};
  bool analysis_ok_{true};
  bool auto_mesh_local_{false};
  CompileConfig config_;
};

}  // namespace lang
}  // namespace taichi
