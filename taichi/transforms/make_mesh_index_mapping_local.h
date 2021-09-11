#pragma once

#include "taichi/ir/pass.h"
#include "taichi/ir/statements.h"

#include <set>

namespace taichi {
namespace lang {

class MakeMeshIndexMappingLocal : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };

  MakeMeshIndexMappingLocal(OffloadedStmt *offload,
                            const CompileConfig &config);

  void fetch_mapping_to_bls(mesh::MeshElementType element_type,
                            mesh::ConvType conv_type);
  void replace_conv_statements(mesh::MeshElementType element_type,
                               mesh::ConvType conv_type);

  static void run(OffloadedStmt *offload,
                  const CompileConfig &config,
                  const std::string &kernel_name);

  const CompileConfig &config;
  std::set<std::pair<mesh::MeshElementType, mesh::ConvType>> mappings{};
  std::size_t bls_offset_in_bytes{0};
  OffloadedStmt *offload{nullptr};
  SNode *snode{nullptr};
  DataType data_type;
  int dtype_size{0};
};

}  // namespace lang
}  // namespace taichi
