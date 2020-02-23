#include "arch.h"

TLANG_NAMESPACE_BEGIN

std::string arch_name(taichi::Tlang::Arch arch) {
  switch (arch) {
#define PER_ARCH(x) \
  case Arch::x:     \
    return #x;      \
    break;
#include "inc/archs.inc.h"
#undef PER_ARCH
    default:
      TI_NOT_IMPLEMENTED
  }
}

// Assuming a processor is either a CPU or a GPU. DSP/TPUs not considered.
bool arch_is_cpu(Arch arch) {
  if (arch == Arch::x64 || arch == Arch::arm64 || arch == Arch::js) {
    return true;
  } else {
    return false;
  }
}

bool arch_is_gpu(Arch arch) {
  return !arch_is_cpu(arch);
}

bool arch_use_host_memory(Arch arch) {
  return arch_is_cpu(arch);
}

TLANG_NAMESPACE_END
