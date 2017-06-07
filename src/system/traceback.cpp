/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <cxxabi.h>
#include <vector>
#include <iostream>
#include <cxxabi.h>
#ifdef __APPLE__
#include <execinfo.h>
#endif
#include <string>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include <mutex>

TC_NAMESPACE_BEGIN

static std::mutex traceback_printer_mutex;

void print_traceback() {
#ifdef __APPLE__
    // Modified based on http://www.nullptr.me/2013/04/14/generating-stack-trace-on-os-x/
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
        sscanf(strs[i], "%*s %s %s %s %*s %d", module_name, addr, function_symbol, &offset);

        int valid_cpp_name = 0;
        //  if this is a C++ library, symbol will be demangled
        //  on success function returns 0
        char *function_name = abi::__cxa_demangle(function_symbol,
                                                  NULL, 0, &valid_cpp_name);

        char stack_frame[4096] = {};
        bool is_valid_cpp_name = (valid_cpp_name == 0);
        sprintf(stack_frame, "* %28s | %7d | %s", module_name, offset, function_name);
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
    printf("                            * Taichi Core - Stack Traceback *                             \n");
    printf("==========================================================================================\n");
    printf("|                       Module |  Offset | Function                                      |\n");
    printf("|----------------------------------------------------------------------------------------|\n");
    for (auto trace: stack_frames) {
        const int function_start = 39;
        const int line_width = 86;
        const int function_width = line_width - function_start - 2;
        int i;
        for (i = 0; i < (int)trace.size(); i++) {
            std::cout << trace[i];
            if (i > function_start + 3 && (i - 3 - function_start) % function_width  == 0) {
                std::cout << " |" << std::endl << " ";
                for (int j = 0; j < function_start; j++) {
                    std::cout << " ";
                }
                std::cout << " | ";
            }
        }
        for (int j = 0; j < function_width + 2 - (i - 3 - function_start) % function_width; j++) {
            std::cout << " ";
        }
        std::cout << "|" << std::endl;
    }
    printf("==========================================================================================\n");
    printf("\n");
#endif
}

TC_NAMESPACE_END
