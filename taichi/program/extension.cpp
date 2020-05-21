#include "extension.h"
//#include "taichi/backends/opengl/opengl_api.h"

#include <unordered_map>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

bool is_supported(Arch arch, Extension ext) {
  static std::unordered_map<Arch, std::unordered_set<Extension>> arch2ext = {
      {Arch::x64, {Extension::sparse, Extension::data64, Extension::adstack}},
      {Arch::arm64, {Extension::sparse, Extension::data64, Extension::adstack}},
      {Arch::cuda, {Extension::sparse, Extension::data64, Extension::adstack}},
      {Arch::metal, {}},
      {Arch::opengl, {}},
  };
  // if (with_opengl_extension_data64())
  // arch2ext[Arch::opengl].insert(Extension::data64); // TODO: singleton
  const auto &exts = arch2ext[arch];
  return exts.find(ext) != exts.end();
}

TLANG_NAMESPACE_END
