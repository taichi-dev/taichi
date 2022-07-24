#include "llvm_codegen_utils.h"

namespace taichi {
namespace lang {

std::string type_name(llvm::Type *type) {
  std::string type_name_str;
  llvm::raw_string_ostream rso(type_name_str);
  type->print(rso, /*IsForDebug=*/false, /*NoDetails=*/true);
  return type_name_str;
}

/*
 * Determine whether two types are the same
 * (a type is a renamed version of the other one) based on the
 * type name. Check recursively if the types are function types.
 *
 * The name of a type imported multiple times is added a suffix starting with a
 * "." following by a number. For example, "RuntimeContext" may be renamed to
 * names like "RuntimeContext.0" and "RuntimeContext.8".
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

void check_func_call_signature(llvm::FunctionType *func_type,
                               llvm::StringRef func_name,
                               std::vector<llvm::Value *> &arglist,
                               llvm::IRBuilder<> *builder) {
  int num_params = func_type->getFunctionNumParams();
  if (func_type->isFunctionVarArg()) {
    TI_ASSERT(num_params <= arglist.size());
  } else {
    TI_ERROR_IF(num_params != arglist.size(),
                "Function \"{}\" requires {} arguments but {} provided",
                std::string(func_name), num_params, arglist.size());
  }

  for (int i = 0; i < num_params; i++) {
    auto required = func_type->getFunctionParamType(i);
    auto provided = arglist[i]->getType();
    /*
     * When importing a module from file, the imported `llvm::Type`s can get
     * conflict with the same type in the llvm::Context. In such scenario,
     * the imported types will be renamed from "original_type" to
     * "original_type.xxx", making them separate types in essence.
     * To make the types of the argument and parameter the same,
     * a pointer cast must be performed.
     */
    if (required != provided) {
      if (is_same_type(required, provided)) {
        arglist[i] = builder->CreatePointerCast(arglist[i], required);
        continue;
      }
      TI_INFO("Function : {}", std::string(func_name));
      TI_INFO("    Type : {}", type_name(func_type));
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
