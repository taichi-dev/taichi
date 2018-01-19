/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/system/threading.h>
#include <csignal>
#include <spdlog/spdlog.h>
#include <taichi/geometry/factory.h>

TC_NAMESPACE_BEGIN

Function11 python_at_exit;

void signal_handler(int signo);

#define TC_REGISTER_SIGNAL_HANDLER(name, handler)                    \
  {                                                                  \
    if (std::signal(name, handler) == SIG_ERR)                       \
      std::printf("Can not register signal handler for" #name "\n"); \
  }

void Logger::set_level(const std::string &level) {
  /*
  trace = 0,
  debug = 1,
  info = 2,
  warn = 3,
  err = 4,
  critical = 5,
  off = 6
  */
  if (level == "trace") {
    spdlog::set_level(spdlog::level::trace);
  } else if (level == "debug") {
    spdlog::set_level(spdlog::level::debug);
  } else if (level == "info") {
    spdlog::set_level(spdlog::level::info);
  } else if (level == "warn") {
    spdlog::set_level(spdlog::level::warn);
  } else if (level == "error") {
    spdlog::set_level(spdlog::level::err);
  } else if (level == "critical") {
    spdlog::set_level(spdlog::level::critical);
  } else if (level == "off") {
    spdlog::set_level(spdlog::level::off);
  } else {
    TC_ERROR(
        "Unknown logging level [{}]. Levels = trace, debug, info, warn, error, "
        "critical, off",
        level);
  }
}

Logger::Logger() {
  console = spdlog::stdout_color_mt("console");
  TC_LOG_SET_PATTERN("[%L %D %X.%e] %v")

  TC_REGISTER_SIGNAL_HANDLER(SIGSEGV, signal_handler);
  TC_REGISTER_SIGNAL_HANDLER(SIGABRT, signal_handler);
  TC_REGISTER_SIGNAL_HANDLER(SIGBUS, signal_handler);
  TC_REGISTER_SIGNAL_HANDLER(SIGFPE, signal_handler);
  spdlog::set_level(spdlog::level::trace);
  TC_TRACE("Taichi core started. Thread ID = {}", PID::get_pid());
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
void Logger::error(const std::string &s, bool raise_signal) {
  console->error(s);
  if (raise_signal) {
    std::raise(SIGABRT);
  }
}
void Logger::critical(const std::string &s, bool raise_signal) {
  console->critical(s);
  if (raise_signal) {
    std::raise(SIGABRT);
  }
}
void Logger::flush() {
  console->flush();
}

Logger logger;

void signal_handler(int signo) {
  logger.error(fmt::format("Received signal {} ({})", signo, strsignal(signo)),
               false);
  TC_FLUSH_LOGGER;
  taichi::print_traceback();
  if (taichi::CoreState::get_instance().trigger_gdb_when_crash) {
    trash(system(fmt::format("sudo gdb -p {}", PID::get_pid()).c_str()));
  }
  if (python_at_exit) {
    TC_INFO("Invoking registered Python at_exit...");
    python_at_exit(0);
    TC_INFO("Python-side at_exit returned.");
  }
  if (taichi::CoreState::get_instance().python_imported) {
    std::string msg =
        fmt::format("Taichi Core Exception: {} ({})", signo, strsignal(signo));
    taichi_raise_assertion_failure_in_python(msg.c_str());
  }
  std::exit(-1);
}

TC_NAMESPACE_END
