#pragma once

#include <sstream>
#include <iostream>

TI_NAMESPACE_BEGIN

struct PythonPrintBuffer {
  /* holds kernel print result before switching back to python */
  std::stringstream ss;
  bool enabled{false};

  template <typename T>
  PythonPrintBuffer &operator<<(const T &t) {
    if (enabled)
      ss << t;
    else
      std::cout << t;
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
