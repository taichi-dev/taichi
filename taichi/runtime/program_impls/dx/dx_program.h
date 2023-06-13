#pragma once

#ifdef TI_WITH_DX11

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/program_impls/gfx/gfx_program.h"

namespace taichi::lang {

class Dx11ProgramImpl : public GfxProgramImpl {
 public:
  Dx11ProgramImpl(CompileConfig &config);

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;
};

}  // namespace taichi::lang

#endif
