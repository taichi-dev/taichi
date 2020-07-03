#include "taichi/backends/metal/env_config.h"

#include "taichi/lang_util.h"
#include "taichi/util/environ_config.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

EnvConfig::EnvConfig() {
  simdgroup_enabled_ =
      get_environ_config("TI_USE_METAL_SIMDGROUP", /*default_value=*/1);
}

const EnvConfig &EnvConfig::instance() {
  static const EnvConfig c;
  return c;
}

}  // namespace metal

TLANG_NAMESPACE_END
