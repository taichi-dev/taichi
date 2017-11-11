/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/logging.h>
#include <taichi/system/threading.h>
#include <signal.h>

TC_NAMESPACE_BEGIN

std::shared_ptr<spdlog::logger> console = spdlog::stdout_color_mt("console");

void handler(int signo) {
  TC_ERR("Received signal {} ({})", signo, strsignal(signo));
  TC_FLUSH_LOGGER;
  taichi::print_traceback();
  std::exit(-1);
}

#define TC_REGISTER_SIGNAL_HANDLER(name, handler)                    \
  {                                                                  \
    if (signal(name, handler) == SIG_ERR)                            \
      std::printf("Can not register signal handler for" #name "\n"); \
  }

class OnStartUp {
 public:
  OnStartUp() {
    TC_REGISTER_SIGNAL_HANDLER(SIGSEGV, handler);
    TC_REGISTER_SIGNAL_HANDLER(SIGABRT, handler);
    TC_REGISTER_SIGNAL_HANDLER(SIGBUS, handler);
    TC_REGISTER_SIGNAL_HANDLER(SIGFPE, handler);
    spdlog::set_level(spdlog::level::trace);
    TC_TRACE("Taichi core started. Thread ID={}", PID::get_pid());
  }
};

static OnStartUp on_start_up;

TC_NAMESPACE_END
