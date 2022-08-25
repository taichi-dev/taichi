#include "taichi/rhi/arch.h"

namespace taichi {

std::string arch_name(Arch arch) {
  switch (arch) {
#define PER_ARCH(x) \
  case Arch::x:     \
    return #x;      \
    break;
#include "taichi/inc/archs.inc.h"

#undef PER_ARCH
    default:
      TI_NOT_IMPLEMENTED
  }
}

Arch arch_from_name(const std::string &arch_name) {
#define PER_ARCH(x)           \
  else if (arch_name == #x) { \
    return Arch::x;           \
  }

  if (false) {
  }
#include "taichi/inc/archs.inc.h"

  else {
    TI_ERROR("Unknown architecture name: {}", arch_name);
  }

#undef PER_ARCH
}

// Assuming a processor is either a CPU or a GPU. DSP/TPUs not considered.
bool arch_is_cpu(Arch arch) {
  if (arch == Arch::x64 || arch == Arch::arm64 || arch == Arch::js ||
      arch == Arch::wasm) {
    return true;
  } else {
    return false;
  }
}

bool arch_is_cuda(Arch arch) {
  return arch == Arch::cuda;
}

bool arch_uses_llvm(Arch arch) {
  return (arch == Arch::x64 || arch == Arch::arm64 || arch == Arch::cuda ||
          arch == Arch::dx12 || arch == Arch::wasm);
}

bool arch_is_gpu(Arch arch) {
  return !arch_is_cpu(arch);
}

Arch host_arch() {
#if defined(TI_ARCH_x64)
  return Arch::x64;
#endif
#if defined(TI_ARCH_ARM)
  return Arch::arm64;
#endif
  TI_NOT_IMPLEMENTED
}

bool arch_use_host_memory(Arch arch) {
  return arch_is_cpu(arch);
}

int default_simd_width(Arch arch) {
  if (arch == Arch::x64) {
    return 8;
  } else if (arch == Arch::cuda) {
    return 32;
  } else if (arch == Arch::arm64) {
    return 4;
  } else {
    TI_NOT_IMPLEMENTED;
    return -1;
  }
}

}  // namespace taichi
