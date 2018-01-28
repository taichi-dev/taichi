/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <vector>
#include <iostream>
#ifdef __APPLE__
#include <execinfo.h>
#include <cxxabi.h>
#endif
#ifdef __linux__
#include <execinfo.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <cxxabi.h>
#endif
#include <string>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include <mutex>

TC_NAMESPACE_BEGIN

static std::mutex traceback_printer_mutex;

TC_EXPORT void print_traceback() {
#ifdef __APPLE__
  // Modified based on
  // http://www.nullptr.me/2013/04/14/generating-stack-trace-on-os-x/
  // TODO: print line number instead of offset
  // (https://stackoverflow.com/questions/8278691/how-to-fix-backtrace-line-number-error-in-c)

  // record stack trace upto 128 frames
  int callstack[128] = {};
  // collect stack frames
  int frames = backtrace((void **)callstack, 128);
  // get the human-readable symbols (mangled)
  char **strs = backtrace_symbols((void **)callstack, frames);
  std::vector<std::string> stack_frames;
  for (int i = 0; i < frames; i++) {
    char function_symbol[1024] = {};
    char module_name[1024] = {};
    int offset = 0;
    char addr[48] = {};
    // split the string, take out chunks out of stack trace
    // we are primarily interested in module, function and address
    sscanf(strs[i], "%*s %s %s %s %*s %d", module_name, addr, function_symbol,
           &offset);

    int valid_cpp_name = 0;
    //  if this is a C++ library, symbol will be demangled
    //  on success function returns 0
    char *function_name =
        abi::__cxa_demangle(function_symbol, NULL, 0, &valid_cpp_name);

    char stack_frame[4096] = {};
    bool is_valid_cpp_name = (valid_cpp_name == 0);
    sprintf(stack_frame, "* %28s | %7d | %s", module_name, offset,
            function_name);
    if (function_name != nullptr)
      free(function_name);

    std::string frameStr(stack_frame);
    stack_frames.push_back(frameStr);
  }
  free(strs);

  // Pretty print the traceback table
  // Exclude this function itself
  stack_frames.erase(stack_frames.begin());
  std::reverse(stack_frames.begin(), stack_frames.end());
  std::lock_guard<std::mutex> guard(traceback_printer_mutex);
  printf("\n");
  printf(
      "                            * Taichi Core - Stack Traceback *           "
      "                  \n");
  printf(
      "========================================================================"
      "==================\n");
  printf(
      "|                       Module |  Offset | Function                     "
      "                 |\n");
  printf(
      "|-----------------------------------------------------------------------"
      "-----------------|\n");
  for (auto trace : stack_frames) {
    const int function_start = 39;
    const int line_width = 86;
    const int function_width = line_width - function_start - 2;
    int i;
    for (i = 0; i < (int)trace.size(); i++) {
      std::cout << trace[i];
      if (i > function_start + 3 &&
          (i - 3 - function_start) % function_width == 0) {
        std::cout << " |" << std::endl << " ";
        for (int j = 0; j < function_start; j++) {
          std::cout << " ";
        }
        std::cout << " | ";
      }
    }
    for (int j = 0;
         j < function_width + 2 - (i - 3 - function_start) % function_width;
         j++) {
      std::cout << " ";
    }
    std::cout << "|" << std::endl;
  }
  printf(
      "========================================================================"
      "==================\n");
  printf("\n");
#elif defined(_WIN64)
// Windows
#else
  // Based on http://man7.org/linux/man-pages/man3/backtrace.3.html
  constexpr int BT_BUF_SIZE = 1024;
  int nptrs;
  void *buffer[BT_BUF_SIZE];
  char **strings;

  nptrs = backtrace(buffer, BT_BUF_SIZE);

  // std::printf("backtrace() returned %d addresses\n", nptrs);

  /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
     would produce similar output to the following: */

  strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }

  fmt::print_colored(fmt::MAGENTA, "***************************\n");
  fmt::print_colored(fmt::MAGENTA, "* Taichi Core Stack Trace *\n");
  fmt::print_colored(fmt::MAGENTA, "***************************\n");

  // j = 0: taichi::print_traceback
  for (int j = 1; j < nptrs; j++) {
    std::string s(strings[j]);
    std::size_t start = s.find("(");
    std::size_t end = s.rfind("+");

    std::string line;

    if (start < end) {
      std::string name = s.substr(start + 1, end - start - 1);

      char *demangled_name_;

      int status = -1;

      demangled_name_ = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);

      if (demangled_name_) {
        name = std::string(demangled_name_);
      }

      std::string prefix = s.substr(0, start);

      line = fmt::format("{}: {}", prefix, name);
      free(demangled_name_);
    } else {
      line = s;
    }
    fmt::print_colored(fmt::MAGENTA, "{}\n", line);
  }
  std::free(strings);
#endif
}

TC_NAMESPACE_END
