#pragma once

#include "taichi/rhi/arch.h"

#include <string>

namespace taichi::lang {

// The Taichi core feature set (dense SNode) should probably be supported by all
// the backends. In addition, each backend can optionally support features in
// the extension set.
//
// The notion of core vs. extension feature set is defined mainly because the
// Metal backend can only support the dense SNodes. We may need to define this
// notion more precisely in the future.

enum class Extension {
#define PER_EXTENSION(x) x,
#include "taichi/inc/extensions.inc.h"

#undef PER_EXTENSION
};

bool is_extension_supported(Arch arch, Extension ext);

}  // namespace taichi::lang
