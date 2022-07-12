#include "llvm_codegen_utils.h"

namespace taichi {
namespace lang {

std::string type_name(llvm::Type *type) {
  std::string type_name_str;
  llvm::raw_string_ostream rso(type_name_str);
  type->print(rso, false, true);
  return type_name_str;
}

/*
 * Determine whether two types are the same
 * (required type is required renamed version of the other one) based on the
 * type name. Check recursively if the types are function types.
 */
bool is_same_type(llvm::Type *required, llvm::Type *provided) {
  if (required == provided) {
    return true;
  }
  if (required->isPointerTy() != provided->isPointerTy()) {
    return false;
  }
  if (required->isPointerTy()) {
    required = required->getPointerElementType();
    provided = provided->getPointerElementType();
  }
  if (required->isFunctionTy() != provided->isFunctionTy()) {
    return false;
  }
  if (required->isFunctionTy()) {
    auto req_func = llvm::dyn_cast<llvm::FunctionType>(required);
    auto prov_func = llvm::dyn_cast<llvm::FunctionType>(provided);
    if (!is_same_type(req_func->getReturnType(), prov_func->getReturnType())) {
      return false;
    }
    if (req_func->getNumParams() != prov_func->getNumParams()) {
      return false;
    }
    for (int j = 0; j < req_func->getNumParams(); j++) {
      if (!is_same_type(req_func->getParamType(j),
                        prov_func->getParamType(j))) {
        return false;
      }
    }
    return true;
  }
  auto req_name = type_name(required);
  auto prov_name = type_name(provided);
  int min_len = std::min(req_name.size(), prov_name.size());
  return req_name.substr(0, min_len) == prov_name.substr(0, min_len);
}

void check_func_call_signature(llvm::Value *func,
                               std::vector<llvm::Value *> &arglist,
                               llvm::IRBuilder<> *builder) {
  llvm::FunctionType *func_type = nullptr;
  if (llvm::Function *fn = llvm::dyn_cast<llvm::Function>(func)) {
    func_type = fn->getFunctionType();
  } else if (auto *call = llvm::dyn_cast<llvm::CallInst>(func)) {
    func_type = llvm::cast_or_null<llvm::FunctionType>(
        func->getType()->getPointerElementType());
  }
  int num_params = func_type->getFunctionNumParams();
  if (func_type->isFunctionVarArg()) {
    TI_ASSERT(num_params <= arglist.size());
  } else {
    TI_ERROR_IF(num_params != arglist.size(),
                "Function \"{}\" requires {} arguments but {} provided",
                std::string(func->getName()), num_params, arglist.size());
  }

  for (int i = 0; i < num_params; i++) {
    auto required = func_type->getFunctionParamType(i);
    auto provided = arglist[i]->getType();
    /*
     * Types in modules imported from files which are not the first appearances
     * are renamed "original_type.xxx", so we have to create a pointer cast to
     * the type in the function parameter list when the types in the function
     * parameter are renamed.
     */
    if (required != provided) {
      if (is_same_type(required, provided)) {
        arglist[i] = builder->CreatePointerCast(arglist[i], required);
        continue;
      }
      TI_INFO("Function : {}", std::string(func->getName()));
      TI_INFO("    Type : {}", type_name(func->getType()));
      if (&required->getContext() != &provided->getContext()) {
        TI_INFO("  parameter {} types are from different contexts", i);
        TI_INFO("    required from context {}",
                (void *)&required->getContext());
        TI_INFO("    provided from context {}",
                (void *)&provided->getContext());
      }
      TI_ERROR("  parameter {} mismatch: required={}, provided={}", i,
               type_name(required), type_name(provided));
    }
  }
}

}  // namespace lang
}  // namespace taichi
