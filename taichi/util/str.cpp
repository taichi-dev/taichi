#include "taichi/util/str.h"

#include <sstream>

#include "taichi/inc/constants.h"

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

std::string format_error_message(const std::string &error_message_template,
                                 const std::function<uint64(int)> &fetcher) {
  std::string error_message_formatted;
  int argument_id = 0;
  for (int i = 0; i < (int)error_message_template.size(); i++) {
    if (error_message_template[i] != '%') {
      error_message_formatted += error_message_template[i];
    } else {
      const auto dtype = error_message_template[i + 1];
      const auto argument = fetcher(argument_id);
      if (dtype == 'd') {
        error_message_formatted += fmt::format(
            "{}", taichi_union_cast_with_different_sizes<int32>(argument));
      } else if (dtype == 'f') {
        error_message_formatted += fmt::format(
            "{}", taichi_union_cast_with_different_sizes<float32>(argument));
      } else {
        TI_ERROR("Data type identifier %{} is not supported", dtype);
      }
      argument_id += 1;
      i++;  // skip the dtype char
    }
  }
  return error_message_formatted;
}

TLANG_NAMESPACE_END
