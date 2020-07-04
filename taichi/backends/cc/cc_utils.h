#pragma once

#include "taichi/lang_util.h"
#include "taichi/common/core.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

TLANG_NAMESPACE_BEGIN
namespace cccp {

inline std::string cc_data_type_name(DataType dt) {
  switch (dt) {
    case DataType::i32:
      return "int";
    case DataType::f32:
      return "float";
    case DataType::f64:
      return "double";
    default:
      TI_ERROR("Unsupported DataType={} on C backend", data_type_name(dt));
  }
}

inline std::string get_func_sym(std::string const &name) {
  return fmt::format("Ti_{}", name);
}

inline std::string get_data_sym(std::string const &name) {
  return fmt::format("ti_{}", name);
}

template <typename... Args>
inline int execute(std::string fmt, Args &&... args) {
  auto cmd = fmt::format(fmt, std::forward<Args>(args)...);
  TI_TRACE("Executing command: {}", cmd);
  int ret = std::system(cmd.c_str());
  TI_TRACE("Command exit status: {}", ret);
  return ret;
}

}  // namespace cccp
TLANG_NAMESPACE_END
