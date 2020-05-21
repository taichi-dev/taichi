#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/backends/metal/data_types.h"
#include "taichi/backends/metal/kernel_util.h"
#include "taichi/backends/metal/kernel_manager.h"
#include "taichi/backends/metal/struct_metal.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

class CodeGen {
 public:
  CodeGen(Kernel *kernel,
          KernelManager *kernel_mgr,
          const CompiledStructs *compiled_structs);

  FunctionType compile();

 private:
  void lower();
  FunctionType gen(const SNode &root_snode, KernelManager *runtime);

  Kernel *const kernel_;
  KernelManager *const kernel_mgr_;
  const CompiledStructs *const compiled_structs_;
  const int id_;
  const std::string taichi_kernel_name_;
};

}  // namespace metal

TLANG_NAMESPACE_END
