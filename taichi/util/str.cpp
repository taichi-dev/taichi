#include "taichi/util/str.h"

#include <sstream>

TLANG_NAMESPACE_BEGIN

std::string c_quoted(std::string const &str) {
  // https://zh.cppreference.com/w/cpp/language/escape
  std::stringstream ss;
  ss << '"';
  for (auto const &c : str) {
    switch (c) {
#define REG_ESC(x, y) \
  case x:             \
    ss << "\\" y;     \
    break;
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
      default:
        ss << c;
    }
  }
#undef REG_ESC
  ss << '"';
  return ss.str();
}

TLANG_NAMESPACE_END
