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

// If |offloaded| is nullptr, this compiles the AST in |kernel|. Otherwise it
// compiles just |offloaded|. These ASTs must have already been lowered at the
// CHI level.
FunctionType compile_to_metal_executable(
    Kernel *kernel,
    KernelManager *kernel_mgr,
    const CompiledStructs *compiled_structs,
    OffloadedStmt *offloaded = nullptr);

}  // namespace metal

TLANG_NAMESPACE_END
