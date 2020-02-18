/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/system/threading.h>
#include <csignal>
#include <spdlog/spdlog.h>
#include <taichi/geometry/factory.h>

TI_NAMESPACE_BEGIN

Function11 python_at_exit;

const auto default_logging_level = "info";

void signal_handler(int signo);

#define TI_REGISTER_SIGNAL_HANDLER(name, handler)                   \
  {                                                                 \
    if (std::signal(name, handler) == SIG_ERR)                      \
      std::printf("Cannot register signal handler for" #name "\n"); \
  }

void Logger::set_level(const std::string &level_name) {
  auto new_level = level_enum_from_string(level_name);
  level = new_level;
  spdlog::set_level((spdlog::level::level_enum)level);
}

int Logger::get_level() {
  return level;
}

bool Logger::is_level_effective(const std::string &level_name) {
  return get_level() <= level_enum_from_string(level_name);
}

int Logger::level_enum_from_string(const std::string &level_name) {
  if (level_name == "trace") {
    return spdlog::level::trace;
  } else if (level_name == "debug") {
    return spdlog::level::debug;
  } else if (level_name == "info") {
    return spdlog::level::info;
  } else if (level_name == "warn") {
    return spdlog::level::warn;
  } else if (level_name == "error") {
    return spdlog::level::err;
  } else if (level_name == "critical") {
    return spdlog::level::critical;
  } else if (level_name == "off") {
    return spdlog::level::off;
  } else {
    TI_ERROR(
        "Unknown logging level [{}]. Levels = trace, debug, info, warn, error, "
        "critical, off",
        level_name);
  }
}

Logger::Logger() {
  console = spdlog::stdout_color_mt("console");
  console->flush_on(spdlog::level::trace);
  TI_LOG_SET_PATTERN("[%L %D %X.%e] %v")

  TI_REGISTER_SIGNAL_HANDLER(SIGSEGV, signal_handler);
  TI_REGISTER_SIGNAL_HANDLER(SIGABRT, signal_handler);
#if !defined(_WIN64)
  TI_REGISTER_SIGNAL_HANDLER(SIGBUS, signal_handler);
#endif
  TI_REGISTER_SIGNAL_HANDLER(SIGFPE, signal_handler);
  set_level_default();
  TI_TRACE("Taichi core started. Thread ID = {}", PID::get_pid());
}

void Logger::set_level_default() {
  set_level(default_logging_level);
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

std::string signal_name(int sig) {
#if !defined(_WIN64)
  return strsignal(sig);
#else
  if (sig == SIGABRT) {
    return "SIGABRT";
  } else if (sig == SIGFPE) {
    return "SIGFPE";
  } else if (sig == SIGILL) {
    return "SIGFPE";
  } else if (sig == SIGSEGV) {
    return "SIGSEGV";
  } else if (sig == SIGTERM) {
    return "SIGTERM";
  } else {
    return "SIGNAL-Unknown";
  }
#endif
}

bool python_at_exit_called = false;

void signal_handler(int signo) {
  logger.error(
      fmt::format("Received signal {} ({})", signo, signal_name(signo)), false);
  TI_FLUSH_LOGGER;
  taichi::print_traceback();
  fmt::print("\n\n\n");
  if (taichi::CoreState::get_instance().trigger_gdb_when_crash) {
#if defined(TI_PLATFORM_LINUX)
    trash(system(fmt::format("sudo gdb -p {}", PID::get_pid()).c_str()));
#endif
  }
  if (python_at_exit && !python_at_exit_called) {
    python_at_exit_called = true;
    TI_INFO("Invoking registered Python at_exit...");
    python_at_exit(0);
    TI_INFO("Python-side at_exit returned.");
  }
  if (taichi::CoreState::get_instance().python_imported) {
    std::string msg = fmt::format("Taichi Core Exception: {} ({})", signo,
                                  signal_name(signo));
#if !defined(TI_AMALGAMATED)
    taichi_raise_assertion_failure_in_python(msg.c_str());
#endif
  }
  std::exit(-1);
}

TI_NAMESPACE_END
