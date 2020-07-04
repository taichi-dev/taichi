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
    default:
      TI_NOT_IMPLEMENTED
  }
}

inline std::string get_sym_name(std::string const &name) {
  return fmt::format("Ti_{}", name);
}

template <typename... Args>
inline int execute(std::string fmt, Args &&... args) {
  auto cmd = fmt::format(fmt, std::forward<Args>(args)...);
  TI_INFO("Executing command: {}", cmd);
  int ret = std::system(cmd.c_str());
  TI_INFO("Command exit status: {}", ret);
  return ret;
}

}  // namespace cccp
TLANG_NAMESPACE_END
