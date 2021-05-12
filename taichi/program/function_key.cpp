#include "taichi/program/function_key.h"

namespace taichi {
namespace lang {

FunctionKey::FunctionKey(const std::string &func_name,
                         int func_id,
                         int instance_id)
    : func_name(func_name), func_id(func_id), instance_id(instance_id) {
}

bool FunctionKey::operator==(const FunctionKey &other_key) const {
  return func_id == other_key.func_id && instance_id == other_key.instance_id;
}

std::string FunctionKey::get_full_name() const {
  return func_name + "_" + std::to_string(func_id) + "_" +
         std::to_string(instance_id);
}

}  // namespace lang
}  // namespace taichi
