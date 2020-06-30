#pragma once

#include "taichi/lang_util.h"
#include "taichi/common/core.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

TLANG_NAMESPACE_BEGIN
namespace cccp {

inline std::string c_quoted(std::string const &str) {
  // https://zh.cppreference.com/w/cpp/language/escape
  std::stringstream ss;
  ss << '"';
  for (auto const &c: str) {
    switch (c) {
#define REG_ESC(x, y) case x: ss << "\\" y; break;
    REG_ESC('\n', "n");
    REG_ESC('\a', "a");
    REG_ESC('\b', "b");
    REG_ESC('\?', "?");
    REG_ESC('\v', "v");
    REG_ESC('\t', "t");
    REG_ESC('\f', "f");
    REG_ESC('\'', "'");
    REG_ESC('\"', "\"");
    REG_ESC('\\', "\\");
    default: ss << c;
    }
  }
  ss << '"';
  return ss.str();
}

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
