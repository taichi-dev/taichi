#include "llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

std::string type_name(llvm::Type *type) {
  std::string type_name_str;
  llvm::raw_string_ostream rso(type_name_str);
  type->print(rso);
  return type_name_str;
}

void check_func_call_signature(llvm::Value *func,
                               std::vector<llvm::Value *> arglist) {
  auto func_type = func->getType()->getPointerElementType();
  int num_params = func_type->getFunctionNumParams();
  if (func_type->isFunctionVarArg()) {
    TI_ASSERT(num_params <= arglist.size());
  } else {
    TI_ASSERT(num_params == arglist.size());
  }

  for (int i = 0; i < num_params; i++) {
    auto required = func_type->getFunctionParamType(i);
    auto provided = arglist[i]->getType();
    if (required != provided) {
      TI_INFO("Function : {}", std::string(func->getName()));
      TI_INFO("    Type : {}", type_name(func->getType()));
      if (&required->getContext() != &provided->getContext()) {
        TI_INFO("  parameter {} types are from different contexts", i);
        TI_INFO("    required from context {}",
                (void *)&required->getContext());
        TI_INFO("    provided from context {}",
                (void *)&provided->getContext());
      }
      TI_INFO("  parameter {} mismatch: required={}, provided={}", i,
              type_name(required), type_name(provided));
      TI_ERROR("Bad function signature.");
    }
  }
}

TLANG_NAMESPACE_END
