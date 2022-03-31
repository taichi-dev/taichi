#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

void run_snode();
void autograd();
void aot_save();
void aot_load();

int main() {
  run_snode();
  autograd();
  aot_save();
  aot_load();
  return 0;
}
