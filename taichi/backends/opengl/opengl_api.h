#pragma once

#include "taichi/common/core.h"

#include <string>
#include <vector>
#include <optional>

#include "opengl_kernel_util.h"
#include "opengl_kernel_launcher.h"

TLANG_NAMESPACE_BEGIN

class Kernel;
class OffloadedStmt;

namespace opengl {

bool initialize_opengl(bool error_tolerance = false);
bool is_opengl_api_available();

#define PER_OPENGL_EXTENSION(x) extern bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

#define TI_OPENGL_REQUIRE(used, x) \
  ([&]() {                         \
    if (opengl_extension_##x) {    \
      used.extension_##x = true;   \
      return true;                 \
    }                              \
    return false;                  \
  })()

class ParallelSize {
  // GLSL: stride < invocation < local work group < 'dispatch'
  // CUDA: stride < thread < block < grid
 public:
  std::optional<size_t> strides_per_thread;
  std::optional<size_t> threads_per_block;

  virtual size_t get_num_strides(GLSLLauncher *launcher) const = 0;
  size_t get_num_threads(GLSLLauncher *launcher) const;
  size_t get_num_blocks(GLSLLauncher *launcher) const;
  virtual size_t get_threads_per_block() const;
  virtual ~ParallelSize();
};

class ParallelSize_ConstRange : public ParallelSize {
  size_t num_strides{1};

 public:
  ParallelSize_ConstRange(size_t num_strides);
  virtual size_t get_num_strides(GLSLLauncher *launcher) const override;
  virtual size_t get_threads_per_block() const override;
  virtual ~ParallelSize_ConstRange() override = default;
};

class ParallelSize_DynamicRange : public ParallelSize {
  bool const_begin;
  bool const_end;
  int range_begin;
  int range_end;

 public:
  ParallelSize_DynamicRange(OffloadedStmt *stmt);
  virtual size_t get_num_strides(GLSLLauncher *launcher) const override;
  virtual ~ParallelSize_DynamicRange() override = default;
};

class ParallelSize_StructFor : public ParallelSize {
 public:
  ParallelSize_StructFor(OffloadedStmt *stmt);
  virtual size_t get_num_strides(GLSLLauncher *launcher) const override;
  virtual ~ParallelSize_StructFor() override = default;
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
           std::unique_ptr<ParallelSize> ps);
  void set_used(const UsedFeature &used);
  int lookup_or_add_string(const std::string &str);
  void launch(Context &ctx, GLSLLauncher *launcher) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
