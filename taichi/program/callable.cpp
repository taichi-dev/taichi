#include "taichi/program/callable.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

Callable::Callable() = default;

Callable::~Callable() = default;

int Callable::insert_arg(const DataType &dt, bool is_array) {
  args.emplace_back(dt->get_compute_type(), is_array);
  return (int)args.size() - 1;
}

int Callable::insert_ret(const DataType &dt) {
  rets.emplace_back(dt->get_compute_type());
  return (int)rets.size() - 1;
}
int Callable::insert_arr_arg(const DataType &dt,
                             int total_dim,
                             std::vector<int> element_shape) {
  args.emplace_back(dt->get_compute_type(), /*is_array=*/true, /*size=*/0,
                    total_dim, element_shape);
  return (int)args.size() - 1;
}

Callable::CurrentCallableGuard::CurrentCallableGuard(Program *program,
                                                     Callable *callable)
    : program_(program) {
  old_callable_ = program->current_callable;
  program->current_callable = callable;
}

Callable::CurrentCallableGuard::~CurrentCallableGuard() {
  program_->current_callable = old_callable_;
}

}  // namespace lang
}  // namespace taichi
