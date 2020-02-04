#pragma once

#include <string>
#include <taichi/common/util.h>
#include <taichi/common.h>

TLANG_NAMESPACE_BEGIN

enum class Arch {
#define PER_ARCH(x) x,
#include "inc/archs.inc.h"
#undef PER_ARCH
};

inline std::string arch_name(Arch arch) {
  switch (arch) {
#define PER_ARCH(x) \
  case Arch::x:     \
    return #x;      \
    break;
#include "inc/archs.inc.h"
#undef PER_ARCH
    default:
      TC_NOT_IMPLEMENTED
  }
}

TLANG_NAMESPACE_END
