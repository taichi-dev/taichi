#pragma once

#include <string>
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

enum class Arch : int {
#define PER_ARCH(x) x,
#include "taichi/inc/archs.inc.h"

#undef PER_ARCH
};

std::string arch_name(Arch arch);

Arch arch_from_name(const std::string &arch);

bool arch_is_cpu(Arch arch);

bool arch_is_gpu(Arch arch);

Arch host_arch();

bool arch_use_host_memory(Arch arch);

int default_simd_width(Arch arch);

TLANG_NAMESPACE_END
