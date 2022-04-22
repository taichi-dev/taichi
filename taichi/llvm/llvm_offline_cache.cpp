#include "llvm_offline_cache.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Module.h"
#include "taichi/ir/transforms.h"

#include "picosha2.h"

namespace taichi {
namespace lang {

static TI_FORCE_INLINE std::vector<std::uint8_t>
get_offline_cache_key_of_compile_config(CompileConfig *config) {
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
    BinaryOutputSerializer &serializer) {
  for (auto &c : snode->ch) {
    get_offline_cache_key_of_snode_impl(c.get(), serializer);
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
  if (snode->physical_type) {
    serializer(snode->physical_type->to_string());
  }
  serializer(snode->dt->to_string());
  serializer(snode->has_ambient);
  if (!snode->ambient_val.dt->is_primitive(PrimitiveTypeID::unknown)) {
    serializer(snode->ambient_val.stringify());
  }
  if (snode->grad_info && !snode->grad_info->is_primal()) {
    if (auto *grad_snode = snode->grad_info->grad_snode()) {
      get_offline_cache_key_of_snode_impl(grad_snode, serializer);
    }
  }
  if (snode->exp_snode) {
    get_offline_cache_key_of_snode_impl(snode->exp_snode, serializer);
  }
  serializer(snode->bit_offset);
  serializer(snode->placing_shared_exp);
  serializer(snode->owns_shared_exponent);
  for (auto s : snode->exponent_users) {
    get_offline_cache_key_of_snode_impl(s, serializer);
  }
  if (snode->currently_placing_exp_snode) {
    get_offline_cache_key_of_snode_impl(snode->currently_placing_exp_snode,
                                        serializer);
  }
  if (snode->currently_placing_exp_snode_dtype) {
    serializer(snode->currently_placing_exp_snode_dtype->to_string());
  }
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
  get_offline_cache_key_of_snode_impl(snode, serializer);
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
    irpass::gen_offline_cache_key(kernel->program, kernel->ir.get(),
                                  &kernel_ast_string);
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
  res.insert(res.begin(), kernel->grad ? 'g' : 'n');
  return res;
}

bool LlvmOfflineCacheFileReader::get_kernel_cache(
    LlvmOfflineCache::KernelCacheData &res,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) {
  res.kernel_key = key;
  std::string filename_prefix = path_ + "/" + key;
  {
    std::string filename = filename_prefix + ".ll";
    llvm::SMDiagnostic err;
    res.owned_module = llvm::parseAssemblyFile(filename, err, llvm_ctx);
    res.module = res.owned_module.get();
    if (!res.module)
      return false;
  }
  {
    std::string filename = filename_prefix + "_otnl.txt";
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open())
      return false;
    while (true) {
      std::string line;
      std::getline(in, line, '\n');
      if (line.empty())
        break;
      std::istringstream iss(line);
      auto &task = res.offloaded_task_list.emplace_back();
      iss >> task.name >> task.block_dim >> task.grid_dim;
    }
  }
  return true;
}

void LlvmOfflineCacheFileWriter::dump() {
  for (auto &[k, v] : data_.kernels) {
    std::string filename_prefix = path_ + "/" + k;
    {
      std::string filename = filename_prefix + ".ll";
      std::ofstream os(filename, std::ios::out | std::ios::binary);
      TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
      llvm::SMDiagnostic err;
      llvm::LLVMContext ctx;
      llvm::raw_os_ostream llvm_os(os);
      if (v.module) {
        mangle_offloaded_task_name(k, v.module, v.offloaded_task_list);
        v.module->print(llvm_os, nullptr);
      } else if (v.owned_module) {
        mangle_offloaded_task_name(k, v.owned_module.get(),
                                   v.offloaded_task_list);
        v.owned_module->print(llvm_os, nullptr);
      } else
        TI_ASSERT(false);
    }
    {
      std::string filename = filename_prefix + "_otnl.txt";
      std::ofstream os(filename, std::ios::out | std::ios::binary);
      TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
      for (const auto &task : v.offloaded_task_list) {
        os << task.name << ' ' << task.block_dim << ' ' << task.grid_dim
           << '\n';
      }
    }
  }
}

void LlvmOfflineCacheFileWriter::mangle_offloaded_task_name(
    const std::string &kernel_key,
    llvm::Module *module,
    std::vector<LlvmOfflineCache::OffloadedTaskCacheData>
        &offloaded_task_list) {
  if (!mangled_) {
    std::size_t cnt = 0;
    for (auto &e : offloaded_task_list) {
      std::string mangled_name = kernel_key + std::to_string(cnt++);
      auto func = module->getFunction(e.name);
      TI_ASSERT(func != nullptr);
      func->setName(mangled_name);
      e.name = mangled_name;
    }
  }
}

}  // namespace lang
}  // namespace taichi
