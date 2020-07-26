#include "extension.h"
//#include "taichi/backends/opengl/opengl_api.h"

#include <unordered_map>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

bool is_extension_supported(Arch arch, Extension ext) {
  static std::unordered_map<Arch, std::unordered_set<Extension>> arch2ext = {
      {Arch::x64,
       {Extension::sparse, Extension::data64, Extension::adstack,
        Extension::assertion, Extension::extfunc}},
      {Arch::arm64,
       {Extension::sparse, Extension::data64, Extension::adstack,
        Extension::assertion}},
      {Arch::cuda,
       {Extension::sparse, Extension::data64, Extension::adstack,
        Extension::bls, Extension::assertion}},
      {Arch::metal, {Extension::adstack}},
      {Arch::opengl, {Extension::extfunc}},
      {Arch::cc, {Extension::data64, Extension::extfunc}},
  };
  // if (with_opengl_extension_data64())
  // arch2ext[Arch::opengl].insert(Extension::data64); // TODO: singleton
  const auto &exts = arch2ext[arch];
  return exts.find(ext) != exts.end();
}

TLANG_NAMESPACE_END
