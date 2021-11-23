#pragma once

#include "taichi/ir/pass.h"
#include "taichi/ir/statements.h"
#include "taichi/analysis/mesh_bls_analyzer.h"

#include <set>

namespace taichi {
namespace lang {

class MakeMeshBlockLocal : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };

  MakeMeshBlockLocal(OffloadedStmt *offload, const CompileConfig &config);

  static void run(OffloadedStmt *offload,
                  const CompileConfig &config,
                  const std::string &kernel_name);

 private:
  void simplify_nested_conversion();
  void gather_candidate_mapping();
  void replace_conv_statements();
  void replace_global_ptrs(SNode *snode);

  void fetch_attr_to_bls(Block *body, Stmt *idx_val, Stmt *mapping_val);
  void push_attr_to_global(Block *body, Stmt *idx_val, Stmt *mapping_val);

  Stmt *create_xlogue(
      Stmt *start_val,
      Stmt *end_val,
      std::function<void(Block * /*block*/, Stmt * /*idx_val*/)> body);
  Stmt *create_cache_mapping(
      Stmt *start_val,
      Stmt *end_val,
      std::function<Stmt *(Block * /*block*/, Stmt * /*idx_val*/)> global_val);

  void fetch_mapping(
      std::function<
          Stmt *(Stmt * /*start_val*/,
                 Stmt * /*end_val*/,
                 std::function<Stmt *(Block * /*block*/,
                                      Stmt * /*idx_val*/)>)/*global_val*/>
          mapping_callback_handler,
      std::function<void(Block * /*body*/,
                         Stmt * /*idx_val*/,
                         Stmt * /*mapping_val*/)> attr_callback_handler);

  const CompileConfig &config;
  OffloadedStmt *offload{nullptr};
  std::set<std::pair<mesh::MeshElementType, mesh::ConvType>> mappings{};
  MeshBLSCaches::Rec rec;

  Block *block;

  std::size_t bls_offset_in_bytes{0};
  std::size_t mapping_bls_offset_in_bytes{0};
  std::unordered_map<SNode *, std::size_t> attr_bls_offset_in_bytes{};

  mesh::MeshElementType element_type;
  mesh::ConvType conv_type;
  SNode *mapping_snode{nullptr};
  DataType mapping_data_type;
  int mapping_dtype_size{0};
};

}  // namespace lang
}  // namespace taichi
