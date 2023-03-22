#include "taichi/codegen/llvm/compiled_kernel_data.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"

namespace taichi::lang {

static std::unique_ptr<CompiledKernelData> new_llvm_compiled_kernel_data() {
  return std::make_unique<llvm::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::llvm_creator =
    new_llvm_compiled_kernel_data;

namespace llvm {

CompiledKernelData::CompiledKernelData(Arch arch, InternalData data)
    : arch_(arch), data_(std::move(data)) {
}

Arch CompiledKernelData::arch() const {
  return arch_;
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  return std::make_unique<CompiledKernelData>(arch_, data_);
}

CompiledKernelData::Err CompiledKernelData::load_impl(
    const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_llvm(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()), data_);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }
  ::llvm::SMDiagnostic err;
  auto ret = ::llvm::parseAssemblyString(file.src_code(), err, llvm_ctx_);
  if (!ret) {  // File not found or Parse failed
    TI_DEBUG("Fail to parse llvm::Module from string: {}",
             err.getMessage().str());
    return Err::kParseSrcCodeFailed;
  }
  data_.compiled_data.module = std::move(ret);
  return Err::kNoError;
}

CompiledKernelData::Err CompiledKernelData::dump_impl(
    CompiledKernelDataFile &file) const {
  file.set_arch(arch_);
  try {
    file.set_metadata(liong::json::print(liong::json::serialize(data_)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  std::string str;
  ::llvm::raw_string_ostream oss(str);
  data_.compiled_data.module->print(oss, /*AAW=*/nullptr);
  file.set_src_code(std::move(str));
  return Err::kNoError;
}

}  // namespace llvm
}  // namespace taichi::lang
