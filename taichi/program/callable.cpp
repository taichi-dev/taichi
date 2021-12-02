#include "taichi/program/callable.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

int Callable::insert_arg(const DataType &dt, bool is_external_array) {
  args.emplace_back(dt->get_compute_type(), is_external_array);
  return (int)args.size() - 1;
}

int Callable::insert_ret(const DataType &dt) {
  rets.emplace_back(dt->get_compute_type());
  return (int)rets.size() - 1;
}
int Callable::insert_arr_arg(const DataType &dt,
                             int total_dim,
                             std::vector<int> element_shapes) {
  args.emplace_back(dt->get_compute_type(), true, /*size=*/0, total_dim,
                    element_shapes);
  return (int)args.size() - 1;
}

Callable::CurrentCallableGuard::CurrentCallableGuard(Program *program,
                                                     Callable *callable)
    : program(program) {
  old_callable = program->current_callable;
  program->current_callable = callable;
}

Callable::CurrentCallableGuard::~CurrentCallableGuard() {
  program->current_callable = old_callable;
}

}  // namespace lang
}  // namespace taichi
