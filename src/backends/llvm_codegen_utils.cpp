#if defined(TLANG_WITH_LLVM)

#include "llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

std::string type_name(llvm::Type *type) {
  std::string type_name_str;
  llvm::raw_string_ostream rso(type_name_str);
  type->print(rso);
  return type_name_str;
}

bool check_func_call_signature(llvm::Value *func,
                               std::vector<Value *> arglist) {
  auto func_type = func->getType()->getPointerElementType();
  int num_params = func_type->getFunctionNumParams();
  TC_ASSERT(num_params == arglist.size());

  for (int i = 0; i < (int)arglist.size(); i++) {
    auto required = func_type->getFunctionParamType(i);
    auto provided = arglist[i]->getType();
    if (required != provided) {
      TC_INFO("Function : {}", std::string(func->getName()));
      TC_INFO("    Type : {}", type_name(func->getType()));
      if (&required->getContext() != &provided->getContext()) {
        TC_INFO("  parameter {} types are from different contexts", i);
        TC_INFO("    required from context {}",
                (void *)&required->getContext());
        TC_INFO("    provided from context {}",
                (void *)&provided->getContext());
      }
      TC_INFO("  parameter {} mismatch: required={}, provided={}", i,
              type_name(required), type_name(provided));
      TC_WARN("Bad function signature.");
      return false;
    }
  }
  return true;
}

TLANG_NAMESPACE_END
#endif
