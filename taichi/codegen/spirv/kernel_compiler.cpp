#include "taichi/codegen/spirv/kernel_compiler.h"

#include "taichi/ir/analysis.h"
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"

namespace taichi::lang {
namespace spirv {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) const {
  auto ir = irpass::analysis::clone(kernel_def.ir.get());
  irpass::compile_to_executable(ir.get(), compile_config, &kernel_def,
                                kernel_def.autodiff_mode,
                                /*ad_use_stack=*/false, compile_config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  TI_TRACE("VK codegen for Taichi kernel={}", kernel_def.name);
  KernelCodegen::Params params;
  params.ti_kernel_name = kernel_def.name;
  params.kernel = &kernel_def;
  params.ir_root = &chi_ir;
  params.compiled_structs = *config_.compiled_struct_data;
  params.arch = compile_config.arch;
  params.caps = device_caps;
  params.enable_spv_opt = compile_config.external_optimization_level > 0;
  spirv::KernelCodegen codegen(params);
  spirv::CompiledKernelData::InternalData internal_data;
  codegen.run(internal_data.metadata.kernel_attribs,
              internal_data.src.spirv_src);
  internal_data.metadata.num_snode_trees = config_.compiled_struct_data->size();
  return std::make_unique<spirv::CompiledKernelData>(compile_config.arch,
                                                     internal_data);
}

}  // namespace spirv
}  // namespace taichi::lang
