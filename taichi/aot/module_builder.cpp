#include "taichi/aot/module_builder.h"
#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend(identifier, kernel);
}

void AotModuleBuilder::add_field(const std::string &identifier,
                                 const SNode *rep_snode,
                                 bool is_scalar,
                                 DataType dt,
                                 std::vector<int> shape,
                                 int row_num,
                                 int column_num) {
  add_field_per_backend(identifier, rep_snode, is_scalar, dt, shape, row_num,
                        column_num);
}

void AotModuleBuilder::add_kernel_template(const std::string &identifier,
                                           const std::string &key,
                                           Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend_tmpl(identifier, key, kernel);
}

bool AotModuleBuilder::all_fields_are_dense_in_container(
    const SNode *container) {
  for (const auto &ch : container->ch) {
    if (ch->type != SNodeType::place) {
      return false;
    }
  }
  const auto *parent = container->parent;
  if (!parent) {
    return false;
  }
  if (parent->type != SNodeType::root) {
    return false;
  }
  return true;
}

void AotModuleBuilder::load(const std::string &output_dir) {
  TI_ERROR("Aot loader not supported");
}

void AotModuleBuilder::dump_graph(std::string output_dir) const {
  const std::string graph_file = fmt::format("{}/graphs.tcb", output_dir);
  write_to_binary_file(graphs_, graph_file);
}

void AotModuleBuilder::add_graph(const std::string &name,
                                 const aot::CompiledGraph &graph) {
  if (graphs_.count(name) != 0) {
    TI_ERROR("Graph {} already exists", name);
  }
  // Handle adding kernels separately.
  for (const auto &dispatch : graph.dispatches) {
    add_compiled_kernel(dispatch.compiled_kernel);
  }
  graphs_[name] = graph;
}
}  // namespace lang
}  // namespace taichi
