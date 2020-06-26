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
  // static_range_for
  int num_groups{1};
  int num_threads{1};
  int threads_per_group{1};
  // dynamic_range_for
  bool const_begin{true};
  bool const_end{true};
  size_t range_begin{0};
  size_t range_end{1};
  // list_struct_for
  bool is_list{false};

  KernelParallelAttrib() = default;
  KernelParallelAttrib(OffloadedStmt *stmt);
  KernelParallelAttrib(int num_threads_);
  size_t calc_num_groups(GLSLLaunchGuard &guard) const;
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

  CompiledProgram(Kernel *kernel);
  ~CompiledProgram();

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           KernelParallelAttrib &&kpa);
  void set_used(const UsedFeature &used);
  int lookup_or_add_string(const std::string &str);
  void launch(Context &ctx, GLSLLauncher *launcher) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
