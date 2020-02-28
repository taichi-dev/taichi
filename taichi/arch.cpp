#include "arch.h"

TLANG_NAMESPACE_BEGIN

std::string arch_name(Arch arch) {
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

Arch arch_from_name(const std::string &arch_name) {
#define PER_ARCH(x)           \
  else if (arch_name == #x) { \
    return Arch::x;           \
  }

  if (false) {
  }
#include "inc/archs.inc.h"
  else {
    TI_ERROR("Unknown architecture name: {}", arch_name);
  }

#undef PER_ARCH
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

int default_simd_width(Arch arch) {
  if (arch == Arch::x64) {
    return 8;
  } else if (arch == Arch::cuda) {
    return 32;
  } else {
    TI_NOT_IMPLEMENTED;
    return -1;
  }
}

TLANG_NAMESPACE_END
