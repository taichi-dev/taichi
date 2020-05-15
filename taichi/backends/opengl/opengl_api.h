#pragma once

#include "taichi/common/core.h"

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
#define PER_OPENGL_EXTENSION(x) extern bool opengl_has_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

struct KernelParallelAttrib {
  int num_groups{1};
  int num_threads{1};
  int threads_per_group{1};
  bool const_begin{true}, const_end{true};
  size_t range_begin{0}, range_end{1};

  KernelParallelAttrib() = default;
  KernelParallelAttrib(OffloadedStmt *stmt);
  KernelParallelAttrib(int num_threads_);
  size_t eval(const void *gtmp) const;
  inline bool is_dynamic() const {
    return num_groups == -1;
  }
};

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
           KernelParallelAttrib &&kpa,
           const UsedFeature &used);
  void launch(Context &ctx, GLSLLauncher *launcher) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
