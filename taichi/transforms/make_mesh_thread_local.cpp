#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/transforms/make_mesh_thread_local.h"

namespace taichi {
namespace lang {

const PassID MakeMeshThreadLocal::id = "MakeMeshThreadLocal";

namespace irpass {

void make_mesh_thread_local_offload(OffloadedStmt *offload,
                                    const CompileConfig &config,
                                    const std::string &kernel_name) {
  if (offload->task_type != OffloadedStmt::TaskType::mesh_for) {
    return;
  }

  std::pair</* owned= */ std::unordered_set<mesh::MeshElementType>,
            /* total= */ std::unordered_set<mesh::MeshElementType>>
      accessed = analysis::gather_mesh_thread_local(offload, config);

  std::size_t tls_offset = offload->tls_size;

  auto data_type = PrimitiveType::u32;  // uint32_t type address
  auto dtype_size = data_type_size(data_type);

  if (offload->tls_prologue == nullptr) {
    offload->tls_prologue = std::make_unique<Block>();
    offload->tls_prologue->parent_stmt = offload;
  }

  if (offload->mesh_prologue == nullptr) {
    offload->mesh_prologue = std::make_unique<Block>();
    offload->mesh_prologue->parent_stmt = offload;
  }

  auto patch_idx =
      offload->tls_prologue->insert(std::make_unique<MeshPatchIndexStmt>(), -1);
  auto one = offload->tls_prologue->insert(
      std::make_unique<ConstStmt>(TypedConstant(data_type, 1)), -1);
  auto patch_idx_1 = offload->tls_prologue->insert(
      std::make_unique<BinaryOpStmt>(BinaryOpType::add, patch_idx, one), -1);

  auto make_thread_local_store =
      [&](mesh::MeshElementType element_type,
          const std::unordered_map<mesh::MeshElementType, SNode *> &offset_,
          std::unordered_map<mesh::MeshElementType, Stmt *> &offset_local,
          std::unordered_map<mesh::MeshElementType, Stmt *> &num_local) {
        const auto offset_tls_offset =
            (tls_offset += (dtype_size - tls_offset % dtype_size) % dtype_size);
        tls_offset += dtype_size;  // allocate storage for the TLS variable

        const auto num_tls_offset =
            (tls_offset += (dtype_size - tls_offset % dtype_size) % dtype_size);
        tls_offset += dtype_size;

        // Step 1:
        // Create thread local storage
        {
          auto offset_ptr =
              offload->tls_prologue->push_back<ThreadLocalPtrStmt>(
                  offset_tls_offset,
                  TypeFactory::get_instance().get_pointer_type(data_type));
          auto num_ptr = offload->tls_prologue->push_back<ThreadLocalPtrStmt>(
              num_tls_offset,
              TypeFactory::get_instance().get_pointer_type(data_type));

          const auto offset_snode = offset_.find(element_type);
          TI_ASSERT(offset_snode != offset_.end());
          auto offset_globalptr = offload->tls_prologue->insert(
              std::make_unique<GlobalPtrStmt>(offset_snode->second,
                                              std::vector<Stmt *>{patch_idx}),
              -1);
          auto offset_load = offload->tls_prologue->insert(
              std::make_unique<GlobalLoadStmt>(offset_globalptr), -1);
          auto offset_1_globalptr = offload->tls_prologue->insert(
              std::make_unique<GlobalPtrStmt>(offset_snode->second,
                                              std::vector<Stmt *>{patch_idx_1}),
              -1);
          auto offset_1_load = offload->tls_prologue->insert(
              std::make_unique<GlobalLoadStmt>(offset_1_globalptr), -1);
          auto num_load = offload->tls_prologue->insert(
              std::make_unique<BinaryOpStmt>(BinaryOpType::sub, offset_1_load,
                                             offset_load),
              -1);

          // TODO: do not use GlobalStore for TLS ptr.
          offload->tls_prologue->push_back<GlobalStoreStmt>(offset_ptr,
                                                            offset_load);
          offload->tls_prologue->push_back<GlobalStoreStmt>(num_ptr, num_load);
        }

        // Step 2:
        // Store TLS mesh_prologue ptr to the offloaded statement
        {
          auto offset_ptr =
              offload->mesh_prologue->push_back<ThreadLocalPtrStmt>(
                  offset_tls_offset,
                  TypeFactory::get_instance().get_pointer_type(data_type));
          auto offset_val =
              offload->mesh_prologue->push_back<GlobalLoadStmt>(offset_ptr);
          auto num_ptr = offload->mesh_prologue->push_back<ThreadLocalPtrStmt>(
              num_tls_offset,
              TypeFactory::get_instance().get_pointer_type(data_type));
          auto num_val =
              offload->mesh_prologue->push_back<GlobalLoadStmt>(num_ptr);

          offset_local.insert(std::pair(element_type, offset_val));
          num_local.insert(std::pair(element_type, num_val));
        }
      };

  for (auto element_type : accessed.first) {
    make_thread_local_store(element_type, offload->mesh->owned_offset,
                            offload->owned_offset_local,
                            offload->owned_num_local);
  }

  for (auto element_type : accessed.second) {
    make_thread_local_store(element_type, offload->mesh->total_offset,
                            offload->total_offset_local,
                            offload->total_num_local);
  }
  offload->tls_size = std::max(std::size_t(1), tls_offset);
}

// This pass should happen after offloading but before lower_access
void make_mesh_thread_local(IRNode *root,
                            const CompileConfig &config,
                            const MakeBlockLocalPass::Args &args) {
  TI_AUTO_PROF;

  // =========================================================================================
  // This pass generates code like this:
  // uint32_t total_vertices_offset = _total_vertices_offset[blockIdx.x];
  // uint32_t total_vertices = _total_vertices_offset[blockIdx.x + 1] -
  // total_vertices_offset;

  // uint32_t total_cells_offset = _total_cells_offset[blockIdx.x];
  // uint32_t total_cells = _total_cells_offset[blockIdx.x + 1] -
  // total_cells_offset;

  // uint32_t owned_cells_offset = _owned_cells_offset[blockIdx.x];
  // uint32_t owned_cells = _owned_cells_offset[blockIdx.x + 1] -
  // owned_cells_offset;
  // =========================================================================================

  if (auto root_block = root->cast<Block>()) {
    for (auto &offload : root_block->statements) {
      make_mesh_thread_local_offload(offload->cast<OffloadedStmt>(), config,
                                     args.kernel_name);
    }
  } else {
    make_mesh_thread_local_offload(root->as<OffloadedStmt>(), config,
                                   args.kernel_name);
  }
  type_check(root, config);
}

}  // namespace irpass
}  // namespace lang
}  // namespace taichi
