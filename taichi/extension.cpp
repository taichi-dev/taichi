#include "extension.h"

#include <unordered_map>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

bool is_supported(Arch arch, Extension ext) {
  static std::unordered_map<Arch, std::unordered_set<Extension>> arch2ext = {
      {Arch::x64, {Extension::sparse, Extension::data64}},
      {Arch::arm64, {Extension::sparse, Extension::data64}},
      {Arch::cuda, {Extension::sparse, Extension::data64}},
      {Arch::metal, {}},
  };
  const auto &exts = arch2ext[arch];
  return exts.find(ext) != exts.end();
}

TLANG_NAMESPACE_END
