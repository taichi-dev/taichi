#include "taichi/program/callable.h"
#include "taichi/ir/type.h"
#include "taichi/program/program.h"

namespace taichi::lang {

Callable::Callable() = default;

Callable::~Callable() = default;

std::vector<int> Callable::insert_scalar_param(const DataType &dt,
                                               const std::string &name) {
  auto p = Parameter(dt->get_compute_type(), /*is_array=*/false);
  p.name = name;
  p.ptype = ParameterType::kScalar;
  return add_parameter(p);
}

int Callable::insert_ret(const DataType &dt) {
  rets.emplace_back(dt->get_compute_type());
  return (int)rets.size() - 1;
}

std::vector<int> Callable::insert_arr_param(const DataType &dt,
                                            int total_dim,
                                            std::vector<int> element_shape,
                                            const std::string &name) {
  auto p = Parameter(dt->get_compute_type(), /*is_array=*/true, false, 0,
                     total_dim, element_shape);
  p.name = name;
  return add_parameter(p);
}

std::vector<int> Callable::insert_ndarray_param(const DataType &dt,
                                                int ndim,
                                                const std::string &name,
                                                bool needs_grad) {
  // Transform ndarray param to a struct type with a pointer to `dt`.
  std::vector<int> element_shape{};
  auto dtype = dt;
  if (dt->is<TensorType>()) {
    element_shape = dt->as<TensorType>()->get_shape();
    dtype = dt->as<TensorType>()->get_element_type();
  }
  // FIXME: we have to use dtype here to scalarization.
  // If we could avoid using parameter_list in codegen it'll be fine
  auto *type = TypeFactory::get_instance().get_ndarray_struct_type(dtype, ndim,
                                                                   needs_grad);
  auto p =
      Parameter(type, /*is_array=*/true, false, 0, ndim + element_shape.size(),
                element_shape, BufferFormat::unknown, needs_grad);
  p.name = name;
  p.ptype = ParameterType::kNdarray;
  return add_parameter(p);
}

std::vector<int> Callable::insert_texture_param(int total_dim,
                                                const std::string &name) {
  // FIXME: we shouldn't abuse is_array for texture parameters
  // FIXME: using rwtexture struct type for texture parameters because C-API
  // does not distinguish between texture and rwtexture.
  auto *type = TypeFactory::get_instance().get_rwtexture_struct_type();
  auto p = Parameter(type, /*is_array=*/true, false, 0, total_dim,
                     std::vector<int>{});
  p.name = name;
  p.ptype = ParameterType::kTexture;
  return add_parameter(p);
}

std::vector<int> Callable::insert_pointer_param(const DataType &dt,
                                                const std::string &name) {
  auto p = Parameter(dt->get_compute_type(), /*is_array=*/true);
  p.name = name;
  return add_parameter(p);
}

std::vector<int> Callable::insert_rw_texture_param(int total_dim,
                                                   BufferFormat format,
                                                   const std::string &name) {
  // FIXME: we shouldn't abuse is_array for texture parameters
  auto *type = TypeFactory::get_instance().get_rwtexture_struct_type();
  auto p = Parameter(type, /*is_array=*/true, false, 0, total_dim,
                     std::vector<int>{}, format);
  p.name = name;
  p.ptype = ParameterType::kRWTexture;
  return add_parameter(p);
}

std::vector<int> Callable::insert_argpack_param_and_push(
    const std::string &name) {
  TI_ASSERT(temp_argpack_stack_.size() == temp_indices_stack_.size() &&
            temp_argpack_name_stack_.size() == temp_indices_stack_.size());
  if (temp_argpack_stack_.size() > 0) {
    temp_indices_stack_.push_back(temp_argpack_stack_.top().size());
  } else {
    temp_indices_stack_.push_back(parameter_list.size());
  }
  temp_argpack_stack_.push(std::vector<Parameter>());
  temp_argpack_name_stack_.push(name);
  return temp_indices_stack_;
}

void Callable::pop_argpack_stack() {
  // Compile argpack members to a struct.
  TI_ASSERT(temp_argpack_stack_.size() > 0 && temp_indices_stack_.size() > 0 &&
            temp_argpack_name_stack_.size() > 0);
  std::vector<Parameter> argpack_params = temp_argpack_stack_.top();
  std::vector<AbstractDictionaryMember> members;
  members.reserve(argpack_params.size());
  for (int i = 0; i < argpack_params.size(); i++) {
    auto &param = argpack_params[i];
    members.push_back(
        {param.is_array && !param.get_dtype()->is<StructType>()
             ? TypeFactory::get_instance().get_pointer_type(param.get_dtype())
             : (const Type *)param.get_dtype(),
         fmt::format("arg_{}_{}", fmt::join(temp_indices_stack_, "_"), i)});
  }
  auto *type_inner =
      TypeFactory::get_instance().get_struct_type(members)->as<StructType>();
  auto *type_pointer = TypeFactory::get_instance().get_pointer_type(
      const_cast<StructType *>(type_inner), false);
  auto *type_outter = TypeFactory::get_instance()
                          .get_struct_type({{type_pointer, "data_ptr"}})
                          ->as<StructType>();
  auto p = Parameter(DataType(type_outter), false, true);
  p.name = temp_argpack_name_stack_.top();
  // Pop stacks
  temp_argpack_stack_.pop();
  temp_indices_stack_.pop_back();
  temp_argpack_name_stack_.pop();
  auto indices = add_parameter(p);
  argpack_types[indices] = type_inner;
}

std::vector<int> Callable::add_parameter(const Parameter &param) {
  TI_ASSERT(temp_argpack_stack_.size() == temp_indices_stack_.size() &&
            temp_argpack_name_stack_.size() == temp_indices_stack_.size());
  if (temp_argpack_stack_.size() == 0) {
    parameter_list.push_back(param);
    auto indices = std::vector<int>{(int)parameter_list.size() - 1};
    nested_parameters[indices] = param;
    return indices;
  }
  temp_argpack_stack_.top().push_back(param);
  std::vector<int> ret = temp_indices_stack_;
  ret.push_back(temp_argpack_stack_.top().size() - 1);
  nested_parameters[ret] = param;
  return ret;
}

void Callable::finalize_rets() {
  std::vector<AbstractDictionaryMember> members;
  members.reserve(rets.size());
  for (int i = 0; i < rets.size(); i++) {
    members.push_back({rets[i].dt, fmt::format("ret_{}", i)});
  }
  auto *type =
      TypeFactory::get_instance().get_struct_type(members)->as<StructType>();
  std::string layout = program->get_kernel_return_data_layout();
  std::tie(ret_type, ret_size) =
      program->get_struct_type_with_data_layout(type, layout);
}

void Callable::finalize_params() {
  TI_ASSERT(temp_argpack_stack_.size() == 0 &&
            temp_indices_stack_.size() == 0 &&
            temp_argpack_name_stack_.size() == 0);
  std::vector<AbstractDictionaryMember> members;
  members.reserve(parameter_list.size());
  for (int i = 0; i < parameter_list.size(); i++) {
    auto &param = parameter_list[i];
    members.push_back(
        {param.is_array && !param.get_dtype()->is<StructType>()
             ? TypeFactory::get_instance().get_pointer_type(param.get_dtype())
             : (const Type *)param.get_dtype(),
         fmt::format("arg_{}", i)});
  }
  auto *type =
      TypeFactory::get_instance().get_struct_type(members)->as<StructType>();
  std::string layout = program->get_kernel_argument_data_layout();
  std::tie(args_type, args_size) =
      program->get_struct_type_with_data_layout(type, layout);
}
}  // namespace taichi::lang
