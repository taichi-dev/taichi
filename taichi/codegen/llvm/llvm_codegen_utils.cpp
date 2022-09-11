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
 * Types like `PhysicalCoordinates` occur in every struct module.
 * When a struct module is copied into a LLVM context,
 * types in the module which already exist in the context are renamed
 * by adding a suffix starting with a "." following by a number.
 * For example, "PhysicalCoordinates" may be renamed to
 * names like "PhysicalCoordinates.0" and "PhysicalCoordinates.8".
 */
bool is_same_type(llvm::Type *a, llvm::Type *b) {
  if (a == b) {
    return true;
  }
  if (a->isPointerTy() != b->isPointerTy()) {
    return false;
  }
  if (a->isPointerTy()) {
    return is_same_type(a->getPointerElementType(), b->getPointerElementType());
  }
  if (a->isFunctionTy() != b->isFunctionTy()) {
    return false;
  }
  if (a->isFunctionTy()) {
    auto req_func = llvm::cast<llvm::FunctionType>(a);
    auto prov_func = llvm::cast<llvm::FunctionType>(b);
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

  auto a_name = type_name(a);
  auto b_name = type_name(b);
  if (a_name.size() > b_name.size()) {
    std::swap(a_name, b_name);
  }
  int len_same = 0;
  while (len_same < a_name.size()) {
    if (a_name[len_same] != b_name[len_same]) {
      break;
    }
    len_same++;
  }
  if (len_same != a_name.size()) {
    // a is xxx.yyy, and b is xxx.zzz, yyy and zzz are numbers
    if (len_same == 0) {
      return false;
    }
    int dot_pos = len_same - 1;
    while (dot_pos && a_name[dot_pos] != '.') {
      dot_pos--;
    }
    if (!dot_pos) {
      return false;
    }
    for (int i = dot_pos + 1; i < a_name.size(); i++) {
      if (!std::isdigit(a_name[i])) {
        return false;
      }
    }
    for (int i = dot_pos + 1; i < b_name.size(); i++) {
      if (!std::isdigit(b_name[i])) {
        return false;
      }
    }
  } else {
    // a is xxx, and b is xxx.yyy, yyy is a number
    TI_ASSERT(len_same != b_name.size());
    if (b_name[len_same] != '.') {
      return false;
    }
    for (int i = len_same + 1; i < b_name.size(); i++) {
      if (!std::isdigit(b_name[i])) {
        return false;
      }
    }
  }
  return true;
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
