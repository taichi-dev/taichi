#include "aot_module_loader_impl.h"
#include "aot_module_builder_impl.h"
#include "aot_graph_data.h"
#include <fstream>
#include <type_traits>

#include "taichi/aot/module_data.h"
#include "taichi/aot/graph_data.h"

namespace taichi {
namespace lang {
namespace directx12 {
namespace {
class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(const aot::CompiledFieldData &field) : field_(field) {
  }

 private:
  aot::CompiledFieldData field_;
};

class AotModuleImpl : public aot::Module {
 public:
  explicit AotModuleImpl(const AotModuleParams &params, Arch device_api_backend)
      : device_api_backend_(device_api_backend) {
    const std::string bin_path =
        fmt::format("{}/metadata_dx12.tcb", params.module_path);
    read_from_binary_file(module_data, bin_path);

    for (auto &[name, compiled_kernel] : module_data.kernels) {
      auto &dxil_codes = module_data.dxil_codes[name];
      auto &tasks = compiled_kernel.tasks;
      for (int i = 0; i < tasks.size(); ++i) {
        auto &task = tasks[i];
        dxil_codes.emplace_back(
            read_dxil_container(params.module_path, task.name));
      }
    }

    // FIXME: enable once write graph to graphs_dx12.tcb.
    // const std::string graph_path =
    //    fmt::format("{}/graphs_dx12.tcb", params.module_path);
    // read_from_binary_file(graphs_, graph_path);
  }

  std::unique_ptr<aot::CompiledGraph> get_graph(
      const std::string &name) override {
    TI_ERROR_IF(graphs_.count(name) == 0, "Cannot find graph {}", name);
    std::vector<aot::CompiledDispatch> dispatches;
    for (auto &dispatch : graphs_[name].dispatches) {
      dispatches.push_back({dispatch.kernel_name, dispatch.symbolic_args,
                            get_kernel(dispatch.kernel_name)});
    }
    aot::CompiledGraph graph{dispatches};
    return std::make_unique<aot::CompiledGraph>(std::move(graph));
  }

  size_t get_root_size() const override {
    return module_data.root_buffer_size;
  }

  // Module metadata
  Arch arch() const override {
    return device_api_backend_;
  }
  uint64_t version() const override {
    TI_NOT_IMPLEMENTED;
  }

 private:
  bool get_field_data_by_name(const std::string &name,
                              aot::CompiledFieldData &field) {
    for (int i = 0; i < module_data.fields.size(); ++i) {
      if (module_data.fields[i].field_name.rfind(name, 0) == 0) {
        field = module_data.fields[i];
        return true;
      }
    }
    return false;
  }

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    if (module_data.kernels.find(name) == module_data.kernels.end())
      return nullptr;
    return std::make_unique<KernelImpl>();
  }

  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &name) override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override {
    aot::CompiledFieldData field;
    if (!get_field_data_by_name(name, field)) {
      TI_DEBUG("Failed to load field {}", name);
      return nullptr;
    }
    return std::make_unique<FieldImpl>(field);
  }

  std::vector<uint8_t> read_dxil_container(const std::string &output_dir,
                                           const std::string &name) {
    const std::string path = fmt::format("{}/{}.dxc", output_dir, name);
    std::vector<uint8_t> source_code;
    std::ifstream fs(path, std::ios_base::binary | std::ios::ate);
    size_t size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    source_code.resize(size / sizeof(uint8_t));
    fs.read((char *)source_code.data(), size);
    fs.close();
    return source_code;
  }

  ModuleDataDX12 module_data;
  Arch device_api_backend_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params,
                                             Arch device_api_backend) {
  AotModuleParams params = std::any_cast<AotModuleParams &>(mod_params);
  return std::make_unique<AotModuleImpl>(params, device_api_backend);
}

}  // namespace directx12
}  // namespace lang
}  // namespace taichi
