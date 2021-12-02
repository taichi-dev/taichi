#pragma once

#include "taichi/lang_util.h"

#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi {
namespace lang {

class Kernel;

namespace spirv {

void lower(Kernel *kernel);

class KernelCodegen {
 public:
  struct Params {
    std::string ti_kernel_name;
    Kernel *kernel;
    std::vector<CompiledSNodeStructs> compiled_structs;
    Device *device;
    bool enable_spv_opt{true};
  };

  explicit KernelCodegen(const Params &params);

  void run(TaichiKernelAttributes &kernel_attribs, std::vector<std::vector<uint32_t>> &generated_spirv);

 private:
  Params params_;
  KernelContextAttributes ctx_attribs_;
};

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
