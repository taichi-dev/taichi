/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/system/traceback.h"

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <mutex>
#include "spdlog/fmt/bundled/color.h"

#if defined(__APPLE__) || (defined(__unix__) && !defined(__linux__)) && \
                              !defined(ANDROID) && !defined(TI_EMSCRIPTENED)
#include <execinfo.h>
#include <cxxabi.h>
#endif
#ifdef _WIN64
#include <intrin.h>
#include <dbghelp.h>

#include "taichi/platform/windows/windows.h"

#pragma comment(lib, "dbghelp.lib")
//  https://gist.github.com/rioki/85ca8295d51a5e0b7c56e5005b0ba8b4
//
//  Debug Helpers
//
// Copyright (c) 2015 - 2017 Sean Farrell <sean.farrell@rioki.org>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

namespace dbg {
void trace(const char *msg, ...) {
  char buff[1024];

  va_list args;
  va_start(args, msg);
  vsnprintf(buff, 1024, msg, args);

  OutputDebugStringA(buff);

  va_end(args);
}

std::string basename(const std::string &file) {
  unsigned int i = file.find_last_of("\\/");
  if (i == std::string::npos) {
    return file;
  } else {
    return file.substr(i + 1);
  }
}

struct StackFrame {
  DWORD64 address;
  std::string name;
  std::string module;
  unsigned int line;
  std::string file;
};

inline std::vector<StackFrame> stack_trace() {
#if _WIN64
  DWORD machine = IMAGE_FILE_MACHINE_AMD64;
#else
  DWORD machine = IMAGE_FILE_MACHINE_I386;
#endif
  HANDLE process = GetCurrentProcess();
  HANDLE thread = GetCurrentThread();

  if (SymInitialize(process, NULL, TRUE) == FALSE) {
    trace("Failed to call SymInitialize.");
    return std::vector<StackFrame>();
  }

  SymSetOptions(SYMOPT_LOAD_LINES);

  CONTEXT context = {};
  context.ContextFlags = CONTEXT_FULL;
  RtlCaptureContext(&context);

#if _WIN64
  STACKFRAME frame = {};
  frame.AddrPC.Offset = context.Rip;
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Offset = context.Rbp;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Offset = context.Rsp;
  frame.AddrStack.Mode = AddrModeFlat;
#else
  STACKFRAME frame = {};
  frame.AddrPC.Offset = context.Eip;
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Offset = context.Ebp;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Offset = context.Esp;
  frame.AddrStack.Mode = AddrModeFlat;
#endif

  bool first = true;

  std::vector<StackFrame> frames;
  while (StackWalk(machine, process, thread, &frame, &context, NULL,
                   SymFunctionTableAccess, SymGetModuleBase, NULL)) {
    StackFrame f = {};
    f.address = frame.AddrPC.Offset;

#if _WIN64
    DWORD64 moduleBase = 0;
#else
    DWORD moduleBase = 0;
#endif

    moduleBase = SymGetModuleBase(process, frame.AddrPC.Offset);

    char moduelBuff[MAX_PATH];
    if (moduleBase &&
        GetModuleFileNameA((HINSTANCE)moduleBase, moduelBuff, MAX_PATH)) {
      f.module = basename(moduelBuff);
    } else {
      f.module = "Unknown Module";
    }
#if _WIN64
    DWORD64 offset = 0;
#else
    DWORD offset = 0;
#endif
    char symbolBuffer[sizeof(IMAGEHLP_SYMBOL) + 255];
    PIMAGEHLP_SYMBOL symbol = (PIMAGEHLP_SYMBOL)symbolBuffer;
    symbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL) + 255;
    symbol->MaxNameLength = 254;

    if (SymGetSymFromAddr(process, frame.AddrPC.Offset, &offset, symbol)) {
      f.name = symbol->Name;
    } else {
      DWORD error = GetLastError();
      trace("Failed to resolve address 0x%X: %u\n", frame.AddrPC.Offset, error);
      f.name = "Unknown Function";
    }

    IMAGEHLP_LINE line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);

    DWORD offset_ln = 0;
    if (SymGetLineFromAddr(process, frame.AddrPC.Offset, &offset_ln, &line)) {
      f.file = line.FileName;
      f.line = line.LineNumber;
    } else {
      DWORD error = GetLastError();
      trace("Failed to resolve line for 0x%X: %u\n", frame.AddrPC.Offset,
            error);
      f.line = 0;
    }

    if (!first) {
      frames.push_back(f);
    }
    first = false;
  }

  SymCleanup(process);

  return frames;
}
}  // namespace dbg
#endif
#if defined(__linux__) && !defined(ANDROID)
#include <execinfo.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <cxxabi.h>
#endif

TI_NAMESPACE_BEGIN

void print_traceback() {
#ifdef __APPLE__
  static std::mutex traceback_printer_mutex;
  // Modified based on
  // http://www.nullptr.me/2013/04/14/generating-stack-trace-on-os-x/
  // TODO: print line number instead of offset
  // (https://stackoverflow.com/questions/8278691/how-to-fix-backtrace-line-number-error-in-c)

  // record stack trace upto 128 frames
  void *callstack[128] = {};
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
  std::reverse(stack_frames.begin(), stack_frames.end());
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
  fmt::print(fg(fmt::color::magenta), "***********************************\n");
  fmt::print(fg(fmt::color::magenta), "* Taichi Compiler Stack Traceback *\n");
  fmt::print(fg(fmt::color::magenta), "***********************************\n");

  std::vector<dbg::StackFrame> stack = dbg::stack_trace();
  for (unsigned int i = 0; i < stack.size(); i++) {
    fmt::print(fg(fmt::color::magenta),
               fmt::format("0x{:x}: ", stack[i].address));
    fmt::print(fg(fmt::color::red), stack[i].name);
    if (stack[i].file != std::string(""))
      fmt::print(fg(fmt::color::magenta),
                 fmt::format("(line {} in {})", stack[i].line, stack[i].file));
    fmt::print(fg(fmt::color::magenta),
               fmt::format(" in {}\n", stack[i].module));
  }
#elif defined(ANDROID)
  // Not supported
  fmt::print(fg(fmt::color::magenta), "***********************************\n");
  fmt::print(fg(fmt::color::magenta), "* Taichi Compiler Stack Traceback *\n");
  fmt::print(fg(fmt::color::magenta), "***********************************\n");
  fmt::print(fg(fmt::color::magenta), "NOT SUPPORTED ON ANDROID\n");
#elif defined(TI_EMSCRIPTENED)
  // Not supported
  fmt::print(fg(fmt::color::magenta), "***********************************\n");
  fmt::print(fg(fmt::color::magenta),
             "* Emscriptened Taichi Compiler Stack Traceback *\n");
  fmt::print(fg(fmt::color::magenta), "***********************************\n");
#else
  // Based on http://man7.org/linux/man-pages/man3/backtrace.3.html
  constexpr int BT_BUF_SIZE = 1024;
  int nptrs;
  void *buffer[BT_BUF_SIZE];
  char **strings;

  nptrs = backtrace(buffer, BT_BUF_SIZE);
  strings = backtrace_symbols(buffer, nptrs);

  if (strings == nullptr) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }

  fmt::print(fg(fmt::color::magenta), "***********************************\n");
  fmt::print(fg(fmt::color::magenta), "* Taichi Compiler Stack Traceback *\n");
  fmt::print(fg(fmt::color::magenta), "***********************************\n");

  // j = 0: taichi::print_traceback
  for (int j = 1; j < nptrs; j++) {
    std::string s(strings[j]);
    std::size_t slash = s.find("/");
    std::size_t start = s.find("(");
    std::size_t end = s.rfind("+");

    std::string line;

    if (slash == 0 && start < end && s[start + 1] != '+') {
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
    fmt::print(fg(fmt::color::magenta), "{}\n", line);
  }
  std::free(strings);
#endif

  fmt::print(
      fg(fmt::color::orange),
      "\nInternal error occurred. Check out this page for possible solutions:\n"
      "https://docs.taichi-lang.org/docs/install\n");
}

TI_NAMESPACE_END
