#include "taichi/runtime/gfx/aot_module_loader_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {
namespace gfx {
namespace {
class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(GfxRuntime *runtime, const aot::CompiledFieldData &field)
      : runtime_(runtime), field_(field) {
  }

 private:
  GfxRuntime *const runtime_;
  aot::CompiledFieldData field_;
};

class AotModuleImpl : public aot::Module {
 public:
  explicit AotModuleImpl(const AotModuleParams &params, Arch device_api_backend)
      : module_path_(params.module_path),
        runtime_(params.runtime),
        device_api_backend_(device_api_backend) {
    std::unique_ptr<io::VirtualDir> dir_alt =
        io::VirtualDir::from_fs_dir(module_path_);
    const io::VirtualDir *dir =
        params.dir == nullptr ? dir_alt.get() : params.dir;

    {
      std::vector<uint8_t> metadata_json{};
      bool succ = dir->load_file("metadata.json", metadata_json) != 0;

      if (!succ) {
        mark_corrupted();
        TI_WARN("'metadata.json' cannot be read");
        return;
      }
      auto json = liong::json::parse(
          (const char *)metadata_json.data(),
          (const char *)(metadata_json.data() + metadata_json.size()));
      liong::json::deserialize(json, ti_aot_data_);
    }

    for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
      auto k = ti_aot_data_.kernels[i];
      std::vector<std::vector<uint32_t>> spirv_sources_codes;
      for (int j = 0; j < k.tasks_attribs.size(); ++j) {
        std::string spirv_path = k.tasks_attribs[j].name + ".spv";

        std::vector<uint32_t> spirv;
        dir->load_file(spirv_path, spirv);

        if (spirv.size() == 0) {
          mark_corrupted();
          TI_WARN("spirv '{}' cannot be read", spirv_path);
          return;
        }
        if (spirv.at(0) != 0x07230203) {
          TI_WARN("spirv '{}' has a incorrect magic number {}", spirv_path,
                  spirv.at(0));
        }
        spirv_sources_codes.emplace_back(std::move(spirv));
      }
      ti_aot_data_.spirv_codes.emplace_back(std::move(spirv_sources_codes));
    }

    {
      std::vector<uint8_t> graphs_json{};
      bool succ = dir->load_file("graphs.json", graphs_json) != 0;

      if (!succ) {
        mark_corrupted();
        TI_WARN("'graphs.json' cannot be read");
        return;
      }

      auto json = liong::json::parse(
          (const char *)graphs_json.data(),
          (const char *)(graphs_json.data() + graphs_json.size()));
      liong::json::deserialize(json, graphs_);
    }
  }

  std::unique_ptr<aot::CompiledGraph> get_graph(
      const std::string &name) override {
    auto it = graphs_.find(name);
    if (it == graphs_.end()) {
      TI_DEBUG("Cannot find graph {}", name);
      return nullptr;
    }

    std::vector<aot::CompiledDispatch> dispatches;
    for (auto &dispatch : it->second.dispatches) {
      dispatches.push_back({dispatch.kernel_name, dispatch.symbolic_args,
                            get_kernel(dispatch.kernel_name)});
    }
    aot::CompiledGraph graph{dispatches};
    return std::make_unique<aot::CompiledGraph>(std::move(graph));
  }

  size_t get_root_size() const override {
    return ti_aot_data_.root_buffer_size;
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
    for (int i = 0; i < ti_aot_data_.fields.size(); ++i) {
      if (ti_aot_data_.fields[i].field_name.rfind(name, 0) == 0) {
        field = ti_aot_data_.fields[i];
        return true;
      }
    }
    return false;
  }

  bool get_kernel_params_by_name(const std::string &name,
                                 GfxRuntime::RegisterParams &kernel) {
    for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
      if (ti_aot_data_.kernels[i].name == name) {
        kernel.kernel_attribs = ti_aot_data_.kernels[i];
        kernel.task_spirv_source_codes = ti_aot_data_.spirv_codes[i];
        // We don't have to store the number of SNodeTree in |ti_aot_data_| yet,
        // because right now we only support a single SNodeTree during AOT.
        // TODO: Support multiple SNodeTrees in AOT.
        kernel.num_snode_trees = 1;
        return true;
      }
    }
    return false;
  }

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    GfxRuntime::RegisterParams kparams;
    if (!get_kernel_params_by_name(name, kparams)) {
      TI_DEBUG("Failed to load kernel {}", name);
      return nullptr;
    }
    return std::make_unique<KernelImpl>(runtime_, std::move(kparams));
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
    return std::make_unique<FieldImpl>(runtime_, field);
  }

  static std::vector<uint32_t> read_spv_file(const std::string &output_dir,
                                             const TaskAttributes &k) {
    const std::string spv_path = fmt::format("{}/{}.spv", output_dir, k.name);
    std::vector<uint32_t> source_code;
    std::ifstream fs(spv_path, std::ios_base::binary | std::ios::ate);
    if (fs.is_open()) {
      size_t size = fs.tellg();
      fs.seekg(0, std::ios::beg);
      source_code.resize(size / sizeof(uint32_t));
      fs.read((char *)source_code.data(), size);
      fs.close();
    }
    return source_code;
  }

  std::string module_path_;
  TaichiAotData ti_aot_data_;
  GfxRuntime *runtime_{nullptr};
  Arch device_api_backend_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params,
                                             Arch device_api_backend) {
  AotModuleParams params = std::any_cast<AotModuleParams &>(mod_params);
  return std::make_unique<AotModuleImpl>(params, device_api_backend);
}

}  // namespace gfx
}  // namespace taichi::lang
