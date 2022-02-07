#include "taichi/backends/metal/aot_module_builder_impl.h"

#include <fstream>

#include "taichi/backends/metal/codegen_metal.h"

namespace taichi {
namespace lang {
namespace metal {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    const CompiledRuntimeModule *compiled_runtime_module,
    const std::vector<CompiledStructs> &compiled_snode_trees,
    const std::unordered_set<const SNode *> &fields,
    BufferMetaData buffer_meta_data)
    : compiled_runtime_module_(compiled_runtime_module),
      compiled_snode_trees_(compiled_snode_trees),
      fields_(fields) {
  buffer_meta_data.root_buffer_size = compiled_snode_trees_[0].root_size;
  ti_aot_data_.metadata = buffer_meta_data;
}

void AotModuleBuilderImpl::write_metal_file(const std::string &dir,
                                            const std::string &filename,
                                            const CompiledKernelData &k) const {
  const std::string mtl_path =
      fmt::format("{}/{}_{}.metal", dir, filename, k.kernel_name);
  std::ofstream fs{mtl_path};
  fs << k.source_code;
  fs.close();
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const std::string bin_path =
      fmt::format("{}/{}_metadata.tcb", output_dir, filename);
  write_to_binary_file(ti_aot_data_, bin_path);
  // The txt file is mostly for debugging purpose.
  const std::string txt_path =
      fmt::format("{}/{}_metadata.txt", output_dir, filename);
  TextSerializer ts;
  ts("taichi aot data", ti_aot_data_);
  ts.write_to_file(txt_path);

  for (const auto &k : ti_aot_data_.kernels) {
    write_metal_file(output_dir, filename, k);
  }

  for (const auto &k : ti_aot_data_.tmpl_kernels) {
    for (auto &ki : k.kernel_tmpl_map) {
      write_metal_file(output_dir, filename, ki.second);
    }
  }
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto compiled = run_codegen(compiled_runtime_module_, compiled_snode_trees_,
                              kernel, &strtab_, /*offloaded=*/nullptr);
  compiled.kernel_name = identifier;
  ti_aot_data_.kernels.push_back(std::move(compiled));
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  const auto *dense_snode = rep_snode->parent;
  TI_ASSERT_INFO(fields_.find(dense_snode) != fields_.end(),
                 "dense_snode: id={} type={}", dense_snode->id,
                 dense_snode->get_node_type_name_hinted());
  const auto &dense_desc =
      compiled_snode_trees_[0].snode_descriptors.at(dense_snode->id);
  CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.is_scalar = is_scalar;
  field_data.dtype = to_metal_type(dt);
  field_data.dtype_name = metal_data_type_name(dt);
  field_data.shape = shape;
  field_data.mem_offset_in_parent = dense_desc.mem_offset_in_parent;
  field_data.row_num = row_num;
  field_data.column_num = column_num;
  ti_aot_data_.fields.push_back(field_data);
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  auto compiled = run_codegen(compiled_runtime_module_, compiled_snode_trees_,
                              kernel, &strtab_, /*offloaded=*/nullptr);
  for (auto &k : ti_aot_data_.tmpl_kernels) {
    if (k.kernel_bundle_name == identifier) {
      k.kernel_tmpl_map.insert(std::make_pair(key, compiled));
      return;
    }
  }
  CompiledKernelTmplData tmpldata;
  tmpldata.kernel_bundle_name = identifier;
  tmpldata.kernel_tmpl_map.insert(std::make_pair(key, compiled));
  ti_aot_data_.tmpl_kernels.push_back(tmpldata);
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
