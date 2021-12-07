#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

namespace taichi {
namespace example {
void run_snode();
void autograd();
void aot_save();

int main() {
  run_snode();
  autograd();
  aot_save();
  return 0;
}
}  // namespace example
}  // namespace taichi
