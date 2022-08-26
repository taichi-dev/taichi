#include "taichi/program/callable.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

Callable::Callable() = default;

Callable::~Callable() = default;

int Callable::insert_scalar_arg(const DataType &dt) {
  args.emplace_back(dt->get_compute_type(), /*is_array=*/false);
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

int Callable::insert_texture_arg(const DataType &dt) {
  // FIXME: we shouldn't abuse is_array for texture args
  args.emplace_back(dt->get_compute_type(), /*is_array=*/true);
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
