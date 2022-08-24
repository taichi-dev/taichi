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
  serializer(config->kernel_profiler);
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
  serializer(config->real_matrix);
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

  std::string autodiff_mode =
      std::to_string(static_cast<std::size_t>(kernel->autodiff_mode));
  picosha2::hash256_one_by_one hasher;
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(kernel_ast_string.begin(), kernel_ast_string.end());
  hasher.process(autodiff_mode.begin(), autodiff_mode.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'T');  // The key must start with a letter
  return res;
}

namespace offline_cache {

constexpr std::size_t offline_cache_key_length = 65;
constexpr std::size_t min_mangled_name_length = offline_cache_key_length + 2;

std::string get_cache_path_by_arch(const std::string &base_path, Arch arch) {
  std::string subdir;
  if (arch_uses_llvm(arch)) {
    subdir = "llvm";
  } else if (arch == Arch::vulkan) {
    subdir = "gfx";
  } else {
    return base_path;
  }
  return taichi::join_path(base_path, subdir);
}

bool enabled_wip_offline_cache(bool enable_hint) {
  // CompileConfig::offline_cache is a global option to enable offline cache on
  // all backends To disable WIP offline cache by default & enable when
  // developing/testing:
  const char *enable_env = std::getenv("TI_WIP_OFFLINE_CACHE");
  return enable_hint && enable_env && std::strncmp("1", enable_env, 1) == 0;
}

std::string mangle_name(const std::string &primal_name,
                        const std::string &key) {
  // Result: {primal_name}{key: char[65]}_{(checksum(primal_name)) ^
  // checksum(key)}
  if (key.size() != offline_cache_key_length) {
    return primal_name;
  }
  std::size_t checksum1{0}, checksum2{0};
  for (auto &e : primal_name) {
    checksum1 += std::size_t(e);
  }
  for (auto &e : key) {
    checksum2 += std::size_t(e);
  }
  return fmt::format("{}{}_{}", primal_name, key, checksum1 ^ checksum2);
}

bool try_demangle_name(const std::string &mangled_name,
                       std::string &primal_name,
                       std::string &key) {
  if (mangled_name.size() < min_mangled_name_length) {
    return false;
  }

  std::size_t checksum{0}, checksum1{0}, checksum2{0};
  auto pos = mangled_name.find_last_of('_');
  if (pos == std::string::npos) {
    return false;
  }
  try {
    checksum = std::stoull(mangled_name.substr(pos + 1));
  } catch (const std::exception &) {
    return false;
  }

  std::size_t i = 0, primal_len = pos - offline_cache_key_length;
  for (i = 0; i < primal_len; ++i) {
    checksum1 += (int)mangled_name[i];
  }
  for (; i < pos; ++i) {
    checksum2 += (int)mangled_name[i];
  }
  if ((checksum1 ^ checksum2) != checksum) {
    return false;
  }

  primal_name = mangled_name.substr(0, primal_len);
  key = mangled_name.substr(primal_len, offline_cache_key_length);
  TI_ASSERT(key.size() == offline_cache_key_length);
  TI_ASSERT(primal_name.size() + key.size() == pos);
  return true;
}

}  // namespace offline_cache

}  // namespace lang
}  // namespace taichi
