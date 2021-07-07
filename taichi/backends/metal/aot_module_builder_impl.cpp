#include "taichi/backends/metal/aot_module_builder_impl.h"

#include <fstream>

#include "taichi/backends/metal/codegen_metal.h"
#include "taichi/system/std_filesystem.h"

namespace taichi {
namespace lang {
namespace metal {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    const CompiledStructs *compiled_structs,
    const BufferMetaData &buffer_meta_data)
    : compiled_structs_(compiled_structs), buffer_meta_data_(buffer_meta_data) {
  ti_aot_data_.metadata = buffer_meta_data;
  
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const stdfs::path dir{output_dir};
  const stdfs::path bin_path = dir / fmt::format("{}_metadata.tcb", filename);
  write_to_binary_file(ti_aot_data_, bin_path.string());
  // The txt file is mostly for debugging purpose.
  const stdfs::path txt_path = dir / fmt::format("{}_metadata.txt", filename);
  TextSerializer ts;
  ts("taichi aot data", ti_aot_data_);
  ts.write_to_file(txt_path.string());

  for (const auto &k : ti_aot_data_.kernels) {
    const stdfs::path mtl_path =
        dir / fmt::format("{}_{}.metal", filename, k.kernel_name);
    std::ofstream fs{mtl_path.string()};
    fs << k.source_code;
    fs.close();
  }

  for (const auto &k : ti_aot_data_.tmpl_kernels) {
<<<<<<< HEAD
    for (auto &ki : k.kernel_tmpl_map) {
      const stdfs::path mtl_path =
          dir / fmt::format("{}_{}.metal", filename, ki.second.kernel_name);
=======
    for (auto &ki: k.kernel_tmpl_map) {
      const fs::path mtl_path = 
        dir / fmt::format("{}_{}.metal", filename, ki.second.kernel_name);
>>>>>>> c596fb80 (dump metal files ok (txt file to fix))
      std::ofstream fs{mtl_path.string()};
      fs << ki.second.source_code;
      fs.close();
    }
  }
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto compiled =
      run_codegen(compiled_structs_, kernel, &strtab_, /*offloaded=*/nullptr);
  compiled.kernel_name = identifier;
  ti_aot_data_.kernels.push_back(std::move(compiled));
}

<<<<<<< HEAD
<<<<<<< HEAD
void AotModuleBuilderImpl::add_per_backend_field(const std::string &identifier,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::pair<int, int> shape,
                                                 int vector_size) {
  CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.is_scalar = is_scalar;
  field_data.dtype = to_metal_type(dt);
  field_data.dtype_name = metal_data_type_name(dt);
  field_data.dimension = {shape.first, shape.second};
  field_data.vector_size = vector_size;
  ti_aot_data_.fields.push_back(field_data);
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  auto compiled =
      run_codegen(compiled_structs_, kernel, &strtab_, /*offloaded=*/nullptr);
  for (auto &k : ti_aot_data_.tmpl_kernels) {
    if (k.kernel_bundle_name == identifier) {
      k.kernel_tmpl_map.insert(std::make_pair(key, compiled));
      return;
    }
  }
=======
=======
void AotModuleBuilderImpl::add_per_backend_field(const std::string &identifier) {
  CompiledFieldData field_data;
  field_data.field_name = identifier;
  ti_aot_data_.fields.push_back(field_data);
}

>>>>>>> f33f5e9d (support key generation without user input)
void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier, 
                                    const std::string &key, 
                                    Kernel *kernel) {
  auto compiled =
      run_codegen(compiled_structs_, kernel, &strtab_, /*offloaded=*/nullptr);
<<<<<<< HEAD
>>>>>>> c596fb80 (dump metal files ok (txt file to fix))
=======
  for (auto &k: ti_aot_data_.tmpl_kernels) {
    if (k.kernel_bundle_name == identifier) {
      k.kernel_tmpl_map.insert(std::make_pair(key, compiled));
      return;
    }
  }
>>>>>>> 1747bbbc (supports nicer api for template args)
  CompiledKernelTmplData tmpldata;
  tmpldata.kernel_bundle_name = identifier;
  tmpldata.kernel_tmpl_map.insert(std::make_pair(key, compiled));
  ti_aot_data_.tmpl_kernels.push_back(tmpldata);
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
