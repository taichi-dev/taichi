#include "taichi/program/callable.h"

namespace taichi {
namespace lang {

int Callable::insert_arg(DataType dt, bool is_external_array) {
  args.emplace_back(dt->get_compute_type(), is_external_array, /*size=*/0);
  return (int)args.size() - 1;
}

int Callable::insert_ret(DataType dt) {
  rets.emplace_back(dt->get_compute_type());
  return (int)rets.size() - 1;
}

}  // namespace lang
}  // namespace taichi
