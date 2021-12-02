#pragma once

#include "taichi/lang_util.h"

#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"

namespace taichi {
namespace lang {

class Kernel;

namespace vulkan {

class VkRuntime;

void lower(Kernel *kernel);

VkRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    VkRuntime *runtime);

// These ASTs must have already been lowered at the CHI level.
FunctionType compile_to_executable(Kernel *kernel, VkRuntime *runtime);

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
