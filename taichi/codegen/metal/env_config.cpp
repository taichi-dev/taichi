#include "taichi/codegen/metal/env_config.h"

#include "taichi/util/lang_util.h"
#include "taichi/util/environ_config.h"

namespace taichi::lang {
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

}  // namespace taichi::lang
