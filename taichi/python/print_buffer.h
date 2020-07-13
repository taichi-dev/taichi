#pragma once

#include <sstream>

TI_NAMESPACE_BEGIN

struct PythonPrintBuffer {
  /* holds kernel print result before switching back to python */
  std::stringstream ss;

  template <typename T>
  PythonPrintBuffer &operator<<(const T &t) {
    ss << t;
    return *this;
  }
  std::string pop_content() {
    auto ret = ss.str();
    ss = std::stringstream();
    return ret;
  }
};

extern PythonPrintBuffer py_cout;

TI_NAMESPACE_END
