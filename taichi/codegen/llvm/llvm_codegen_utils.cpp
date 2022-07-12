#include "llvm_codegen_utils.h"

namespace taichi {
namespace lang {

std::string type_name(llvm::Type *type) {
  std::string type_name_str;
  llvm::raw_string_ostream rso(type_name_str);
  type->print(rso);
  return type_name_str;
}


/*
 * Determine whether two types are the same
 * (a type is a renamed version of the other one) based on the type name
 */
bool is_same_type(llvm::Type *a, llvm::Type *b) {
  if (a == b) {
    return true;
  }
  auto a_name = type_name(a);
  auto b_name = type_name(b);
  int min_len = std::min(a_name.size(), b_name.size()) - 1;
  return a_name.substr(0, min_len) == b_name.substr(0, min_len);
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
     * are renamed "original_type.xxx", so we have to create a pointer cast to the
     * type in the function parameter list when the types in the function parameter
     * are renamed.
     */
    if (required != provided) {
      bool is_same = true;
      if (required->isPointerTy() && required->getPointerElementType()->isFunctionTy()) {
        auto req_func = llvm::dyn_cast<llvm::FunctionType>(required->getPointerElementType());
        auto prov_func = llvm::dyn_cast<llvm::FunctionType>(provided->getPointerElementType());
        if (req_func->getNumParams() != prov_func->getNumParams()) {
          is_same = false;
        }
        for (int j = 0; is_same && j < req_func->getNumParams(); j++) {
          if (!is_same_type(req_func->getParamType(j), prov_func->getParamType(j))) {
            is_same = false;
          }
        }
      } else {
        is_same = is_same_type(required, provided);
      }
      if (is_same) {
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
