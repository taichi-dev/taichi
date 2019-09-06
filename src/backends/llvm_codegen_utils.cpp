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
      TC_INFO("Function type: {}", type_name(func->getType()));
      TC_INFO("  parameter {} mismatch: required={}, provided={}", i,
              type_name(required), type_name(provided));
      TC_WARN("Bad function signature.");
      return false;
    }
  }
  return true;
}

TLANG_NAMESPACE_END
