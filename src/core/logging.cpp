/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/system/threading.h>
#include <signal.h>
#include <spdlog/spdlog.h>

TC_NAMESPACE_BEGIN

void signal_handler(int signo) {
  TC_ERROR("Received signal {} ({})", signo, strsignal(signo));
  TC_FLUSH_LOGGER;
  taichi::print_traceback();
  std::exit(-1);
}

#define TC_REGISTER_SIGNAL_HANDLER(name, handler)                    \
  {                                                                  \
    if (signal(name, handler) == SIG_ERR)                            \
      std::printf("Can not register signal handler for" #name "\n"); \
  }

Logger::Logger() {
  console = spdlog::stdout_color_mt("console");

  TC_REGISTER_SIGNAL_HANDLER(SIGSEGV, signal_handler);
  TC_REGISTER_SIGNAL_HANDLER(SIGABRT, signal_handler);
  TC_REGISTER_SIGNAL_HANDLER(SIGBUS, signal_handler);
  TC_REGISTER_SIGNAL_HANDLER(SIGFPE, signal_handler);
  spdlog::set_level(spdlog::level::trace);
  TC_TRACE("Taichi core started. Thread ID={}", PID::get_pid());
}

void Logger::trace(const std::string &s) {
  console->trace(s);
}

void Logger::debug(const std::string &s) {
  console->debug(s);
}

void Logger::info(const std::string &s) {
  console->info(s);
}

void Logger::warn(const std::string &s) {
  console->warn(s);
}
void Logger::error(const std::string &s) {
  console->error(s);
}
void Logger::critical(const std::string &s) {
  console->critical(s);
}
void Logger::flush() {
  console->flush();
}

Logger logger;

TC_NAMESPACE_END
