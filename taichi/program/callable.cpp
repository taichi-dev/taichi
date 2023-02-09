#include "taichi/program/callable.h"
#include "taichi/program/program.h"

namespace taichi::lang {

Callable::Callable() = default;

Callable::~Callable() = default;

int Callable::insert_scalar_param(const DataType &dt) {
  parameter_list.emplace_back(dt->get_compute_type(), /*is_array=*/false);
  return (int)parameter_list.size() - 1;
}

int Callable::insert_ret(const DataType &dt) {
  rets.emplace_back(dt->get_compute_type());
  return (int)rets.size() - 1;
}

int Callable::insert_arr_param(const DataType &dt,
                               int total_dim,
                               std::vector<int> element_shape) {
  parameter_list.emplace_back(dt->get_compute_type(), /*is_array=*/true,
                              /*size=*/0, total_dim, element_shape);
  return (int)parameter_list.size() - 1;
}

int Callable::insert_texture_param(const DataType &dt) {
  // FIXME: we shouldn't abuse is_array for texture parameters
  parameter_list.emplace_back(dt->get_compute_type(), /*is_array=*/true);
  return (int)parameter_list.size() - 1;
}

void Callable::finalize_rets() {
  if (rets.empty()) {
    return;
  }
  std::vector<StructMember> members;
  members.reserve(rets.size());
  for (int i = 0; i < rets.size(); i++) {
    members.push_back({rets[i].dt, fmt::format("ret_{}", i)});
  }
  ret_type =
      TypeFactory::get_instance().get_struct_type(members)->as<StructType>();
}
}  // namespace taichi::lang
