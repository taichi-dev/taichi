#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

// Metal specific config inferred from the environment.
class EnvConfig {
 public:
  // Set TI_USE_METAL_SIMDGROUP=0 to disable SIMD group.
  // This is an ad-hoc thing. Apple claims that SIMD group is supported in
  // MSL 2.0, which isn't the case. According to my test, it's available in
  // MSL 2.1. Since MSL 2.1 is released since macOS 10.14, we expect the major
  // users would be able to use this feature.
  inline bool is_simdgroup_enabled() const {
    return simdgroup_enabled_;
  }

  static const EnvConfig &instance();

 private:
  EnvConfig();

  bool simdgroup_enabled_;
};

}  // namespace metal

TLANG_NAMESPACE_END
