#pragma once

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/program/callable.h"

#include "llvm/IR/LLVMContext.h"

namespace taichi::lang {

namespace LLVM {

class CompiledKernelData : public lang::CompiledKernelData {
 public:
  struct InternalData {
    std::vector<std::pair<std::vector<int>, Callable::Parameter>> args;
    std::vector<Callable::Ret> rets;
    LLVMCompiledKernel compiled_data;

    const StructType *ret_type = nullptr;
    size_t ret_size{0};

    const StructType *args_type = nullptr;
    size_t args_size{0};

    TI_IO_DEF(args,
              rets,
              compiled_data,
              ret_type,
              ret_size,
              args_type,
              args_size);

    InternalData() = default;

    InternalData(const InternalData &o)
        : args(o.args),
          rets(o.rets),
          compiled_data(o.compiled_data.clone()),
          ret_type(o.ret_type),
          ret_size(o.ret_size),
          args_type(o.args_type),
          args_size(o.args_size) {
    }

    InternalData(InternalData &&o) = default;
  };

  CompiledKernelData() = default;
  CompiledKernelData(Arch arch, InternalData data);

  Arch arch() const override;
  std::unique_ptr<lang::CompiledKernelData> clone() const override;

  Err check() const override;

  const InternalData &get_internal_data() const {
    return data_;
  }

 protected:
  Err load_impl(const CompiledKernelDataFile &file) override;
  Err dump_impl(CompiledKernelDataFile &file) const override;

 private:
  llvm::LLVMContext llvm_ctx_;
  Arch arch_;
  InternalData data_;
};

}  // namespace LLVM

}  // namespace taichi::lang
