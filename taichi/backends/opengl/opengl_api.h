#pragma once

#include "taichi/common/core.h"
#include "taichi/ir/transforms.h"

#include <string>
#include <vector>
#include <optional>

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/backends/opengl/opengl_kernel_launcher.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

class Kernel;
class OffloadedStmt;

namespace opengl {

bool initialize_opengl(bool error_tolerance = false);
bool is_opengl_api_available();

#define PER_OPENGL_EXTENSION(x) extern bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

extern int opengl_threads_per_block;

#define TI_OPENGL_REQUIRE(used, x) \
  ([&]() {                         \
    if (opengl_extension_##x) {    \
      used.extension_##x = true;   \
      return true;                 \
    }                              \
    return false;                  \
  })()

struct CompiledKernel;

struct CompiledProgram {
  struct Impl;
  std::unique_ptr<Impl> impl;

  // disscussion:
  // https://github.com/taichi-dev/taichi/pull/696#issuecomment-609332527
  CompiledProgram(CompiledProgram &&) = default;
  CompiledProgram &operator=(CompiledProgram &&) = default;

  CompiledProgram(Kernel *kernel, Device *device);
  ~CompiledProgram();

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           int num_workgrous,
           int workgroup_size,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access =
               nullptr);
  void set_used(const UsedFeature &used);
  int lookup_or_add_string(const std::string &str);
  void launch(Context &ctx, GLSLLauncher *launcher) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
