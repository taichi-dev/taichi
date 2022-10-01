#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/runtime/metal/data_types.h"
#include "taichi/runtime/metal/kernel_manager.h"
#include "taichi/runtime/metal/kernel_utils.h"
#include "taichi/codegen/metal/struct_metal.h"

namespace taichi::lang {
namespace metal {

CompiledKernelData run_codegen(
    const CompiledRuntimeModule *compiled_runtime_module,
    const std::vector<CompiledStructs> &compiled_snode_trees,
    Kernel *kernel,
    OffloadedStmt *offloaded);

FunctionType compiled_kernel_to_metal_executable(
    const CompiledKernelData &compiled_kernel,
    KernelManager *kernel_mgr);

}  // namespace metal
}  // namespace taichi::lang
