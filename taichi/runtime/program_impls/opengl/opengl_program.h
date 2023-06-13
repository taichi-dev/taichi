#pragma once

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/program/program_impl.h"
#include "taichi/runtime/program_impls/gfx/gfx_program.h"

namespace taichi::lang {

class OpenglProgramImpl : public GfxProgramImpl {
 public:
  explicit OpenglProgramImpl(CompileConfig &config);

  void finalize() override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;
};

}  // namespace taichi::lang
