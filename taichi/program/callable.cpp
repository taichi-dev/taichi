#include "taichi/program/callable.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

int Callable::insert_arg(const PrimitiveTypeID &ptid, bool is_external_array) {
  args.emplace_back(ptid->get_compute_type(), is_external_array, /*size=*/0);
  return (int)args.size() - 1;
}

int Callable::insert_ret(const PrimitiveTypeID &ptid) {
  rets.emplace_back(ptid->get_compute_type());
  return (int)rets.size() - 1;
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
