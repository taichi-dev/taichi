#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

void run_snode();
void autograd();
void aot_save(taichi::Arch arch);

int main() {
  run_snode();
  autograd();
#ifdef TI_WITH_VULKAN
  aot_save(taichi::Arch::vulkan);
#endif
#ifdef TI_WITH_DX12
  aot_save(taichi::Arch::dx12);
#endif
  return 0;
}
