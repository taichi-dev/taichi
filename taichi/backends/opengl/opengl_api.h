#pragma once

#include "taichi/common/util.h"

#include <string>
#include <vector>
#include <optional>

#include "opengl_kernel_launcher.h"

TLANG_NAMESPACE_BEGIN

class Kernel;
class OffloadedStmt;

namespace opengl {

bool initialize_opengl(bool error_tolerance = false);
bool is_opengl_api_available();
int opengl_get_threads_per_group();
#define PER_OPENGL_EXTENSION(x) extern bool opengl_has_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

struct RangeSizeEvaluator_ {
  bool const_begin, const_end;
  size_t begin, end;
  size_t gl_threads_per_group;

  RangeSizeEvaluator_(OffloadedStmt *stmt);
  size_t eval(const void *gtmp);
};

using RangeSizeEvaluator = std::optional<RangeSizeEvaluator_>;

struct CompiledProgram {
  struct Impl;
  std::unique_ptr<Impl> impl;

  // disscussion:
  // https://github.com/taichi-dev/taichi/pull/696#issuecomment-609332527
  CompiledProgram(CompiledProgram &&) = default;
  CompiledProgram &operator=(CompiledProgram &&) = default;

  CompiledProgram(Kernel *kernel, size_t gtmp_size);
  ~CompiledProgram();

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           int num_groups,
           RangeSizeEvaluator rse,
           const UsedFeature &used);
  void launch(Context &ctx, GLSLLauncher *launcher) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
