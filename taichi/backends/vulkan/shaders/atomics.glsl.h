// clang-format on
#include "taichi/backends/vulkan/shaders/prologue.h"

#ifndef TI_INSIDE_VULKAN_CODEGEN
static_assert(false, "do not include");
#endif  // TI_INSIDE_VULKAN_CODEGEN

#define VULKAN_BEGIN_CODE_DEF constexpr auto kVulkanAtomicsSourceCode =
#define VULKAN_END_CODE_DEF ;

// clang-format off
VULKAN_BEGIN_CODE_DEF
STR(
// TODO: don't duplicate, pass in pointer
float fatomicAdd_root_buffer(int addr, float data) {
  int old_val = 0;
  int new_val = 0;
  int cas_val = 0;
  int ok = 0;
  while (ok == 0) {
    old_val = root_buffer[addr];
    new_val = floatBitsToInt(intBitsToFloat(old_val) + data);
    cas_val = atomicCompSwap(root_buffer[addr], old_val, new_val);
    ok = int(cas_val == old_val);
  }
  return intBitsToFloat(old_val);
}

float fatomicAdd_global_tmps_buffer(int addr, float data) {
  int old_val = 0;
  int new_val = 0;
  int cas_val = 0;
  int ok = 0;
  while (ok == 0) {
    old_val = global_tmps_buffer[addr];
    new_val = floatBitsToInt(intBitsToFloat(old_val) + data);
    cas_val = atomicCompSwap(global_tmps_buffer[addr], old_val, new_val);
    ok = int(cas_val == old_val);
  }
  return intBitsToFloat(old_val);
}
)
VULKAN_END_CODE_DEF
// clang-format on
