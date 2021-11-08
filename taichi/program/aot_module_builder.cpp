#include "taichi/program/aot_module_builder.h"
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

int AotModuleBuilder::find_children_id(const SNode *snode) {
  auto parent = snode->parent;
  for (int i = 0; i < parent->ch.size(); i++) {
    if (parent->ch[i].get() == snode)
      return i;
  }
  TI_ERROR("Child not found in parent!");
}

}  // namespace lang
}  // namespace taichi
