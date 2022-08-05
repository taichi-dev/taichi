#include "offline_cache_util.h"

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel.h"

#include "picosha2.h"

#include <vector>

namespace taichi {
namespace lang {

static std::vector<std::uint8_t> get_offline_cache_key_of_compile_config(
    CompileConfig *config) {
  TI_ASSERT(config);
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(config->arch);
  serializer(config->debug);
  serializer(config->cfg_optimization);
  serializer(config->check_out_of_bound);
  serializer(config->opt_level);
  serializer(config->external_optimization_level);
  serializer(config->packed);
  serializer(config->move_loop_invariant_outside_if);
  serializer(config->demote_dense_struct_fors);
  serializer(config->advanced_optimization);
  serializer(config->constant_folding);
  serializer(config->fast_math);
  serializer(config->flatten_if);
  serializer(config->make_thread_local);
  serializer(config->make_block_local);
  serializer(config->detect_read_only);
  serializer(config->default_fp->to_string());
  serializer(config->default_ip.to_string());
  if (arch_is_cpu(config->arch)) {
    serializer(config->default_cpu_block_dim);
    serializer(config->cpu_max_num_threads);
  } else if (arch_is_gpu(config->arch)) {
    serializer(config->default_gpu_block_dim);
    serializer(config->gpu_max_reg);
    serializer(config->saturating_grid_dim);
    serializer(config->cpu_max_num_threads);
  }
  serializer(config->ad_stack_size);
  serializer(config->default_ad_stack_size);
  serializer(config->random_seed);
  if (config->arch == Arch::cc) {
    serializer(config->cc_compile_cmd);
    serializer(config->cc_link_cmd);
  } else if (config->arch == Arch::opengl) {
    serializer(config->allow_nv_shader_extension);
    serializer(config->use_gles);
  }
  serializer(config->make_mesh_block_local);
  serializer(config->optimize_mesh_reordered_mapping);
  serializer(config->mesh_localize_to_end_mapping);
  serializer(config->mesh_localize_from_end_mapping);
  serializer(config->mesh_localize_all_attr_mappings);
  serializer(config->demote_no_access_mesh_fors);
  serializer(config->experimental_auto_mesh_local);
  serializer(config->auto_mesh_local_default_occupacy);
  serializer.finalize();

  return serializer.data;
}

static void get_offline_cache_key_of_snode_impl(
    SNode *snode,
    BinaryOutputSerializer &serializer,
    std::unordered_set<int> &visited) {
  if (auto iter = visited.find(snode->id); iter != visited.end()) {
    serializer(snode->id);  // Use snode->id as placeholder to identify a snode
    return;
  }

  visited.insert(snode->id);
  for (auto &c : snode->ch) {
    get_offline_cache_key_of_snode_impl(c.get(), serializer, visited);
  }
  for (int i = 0; i < taichi_max_num_indices; ++i) {
    auto &extractor = snode->extractors[i];
    serializer(extractor.num_elements_from_root);
    serializer(extractor.shape);
    serializer(extractor.acc_shape);
    serializer(extractor.num_bits);
    serializer(extractor.acc_offset);
    serializer(extractor.active);
  }
  serializer(snode->index_offsets);
  serializer(snode->num_active_indices);
  serializer(snode->physical_index_position);
  serializer(snode->id);
  serializer(snode->depth);
  serializer(snode->name);
  serializer(snode->num_cells_per_container);
  serializer(snode->total_num_bits);
  serializer(snode->total_bit_start);
  serializer(snode->chunk_size);
  serializer(snode->cell_size_bytes);
  serializer(snode->offset_bytes_in_parent_cell);
  serializer(snode->dt->to_string());
  serializer(snode->has_ambient);
  if (!snode->ambient_val.dt->is_primitive(PrimitiveTypeID::unknown)) {
    serializer(snode->ambient_val.stringify());
  }
  if (snode->grad_info && !snode->grad_info->is_primal()) {
    if (auto *adjoint_snode = snode->grad_info->adjoint_snode()) {
      get_offline_cache_key_of_snode_impl(adjoint_snode, serializer, visited);
    }
    if (auto *dual_snode = snode->grad_info->dual_snode()) {
      get_offline_cache_key_of_snode_impl(dual_snode, serializer, visited);
    }
  }
  if (snode->physical_type) {
    serializer(snode->physical_type->to_string());
  }
  serializer(snode->id_in_bit_struct);
  serializer(snode->is_bit_level);
  serializer(snode->is_path_all_dense);
  serializer(snode->node_type_name);
  serializer(snode->type);
  serializer(snode->_morton);
  serializer(snode->get_snode_tree_id());
}

std::string get_hashed_offline_cache_key_of_snode(SNode *snode) {
  TI_ASSERT(snode);

  BinaryOutputSerializer serializer;
  serializer.initialize();
  {
    std::unordered_set<int> visited;
    get_offline_cache_key_of_snode_impl(snode, serializer, visited);
  }
  serializer.finalize();

  picosha2::hash256_one_by_one hasher;
  hasher.process(serializer.data.begin(), serializer.data.end());
  hasher.finish();

  return picosha2::get_hash_hex_string(hasher);
}

std::string get_hashed_offline_cache_key(CompileConfig *config,
                                         Kernel *kernel) {
  std::string kernel_ast_string;
  if (kernel) {
    std::ostringstream oss;
    gen_offline_cache_key(kernel->program, kernel->ir.get(), &oss);
    kernel_ast_string = oss.str();
  }

  std::vector<std::uint8_t> compile_config_key;
  if (config) {
    compile_config_key = get_offline_cache_key_of_compile_config(config);
  }

  picosha2::hash256_one_by_one hasher;
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(kernel_ast_string.begin(), kernel_ast_string.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(),
             kernel->autodiff_mode != AutodiffMode::kNone ? 'g' : 'n');
  return res;
}

}  // namespace lang
}  // namespace taichi
