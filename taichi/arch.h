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

std::string arch_name(Arch arch);

Arch arch_from_name(const std::string &arch);

bool arch_is_cpu(Arch arch);

bool arch_is_gpu(Arch arch);

bool arch_use_host_memory(Arch arch);

TLANG_NAMESPACE_END
