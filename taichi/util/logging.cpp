/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/core.h"
#include "taichi/system/threading.h"
#include <csignal>
#include "taichi/python/export.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

TI_NAMESPACE_BEGIN

std::function<void(int)> python_at_exit;

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
  TI_LOG_SET_PATTERN("%^[%L %D %X.%e] %v%$");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const std::string &e) {
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

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

void Logger::error(const std::string &s, bool raise_exception) {
  console->error(s);
  fmt::print("\n\n");
  print_traceback();
  if (taichi::CoreState::get_instance().trigger_gdb_when_crash) {
#if defined(TI_PLATFORM_LINUX)
    trash(system(fmt::format("sudo gdb -p {}", PID::get_pid()).c_str()));
#endif
  }
  if (raise_exception)
    throw s;
}

void Logger::critical(const std::string &s) {
  Logger::error(s);  // simply forward to Logger::error since we actually never
                     // use TI_CRITICAL
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
  // It seems that there's no way to pass exception to Python in signal handlers?
  // @archibate found that in fact there are such solution:
  // https://docs.python.org/3/library/faulthandler.html#module-faulthandler
  auto sig_name = signal_name(signo);
  logger.error(fmt::format("Received signal {} ({})", signo, sig_name), false);
  exit(-1);
  /*
  if (python_at_exit && !python_at_exit_called) {
    python_at_exit_called = true;
    TI_INFO("Invoking registered Python at_exit...");
    python_at_exit(0);
    TI_INFO("Python-side at_exit returned.");
  }
  if (taichi::CoreState::get_instance().python_imported) {
    std::string msg = fmt::format("Taichi Core Exception: {} ({})", signo,
                                  signal_name(signo));
    taichi_raise_assertion_failure_in_python(msg.c_str());
  }
  */
  TI_UNREACHABLE
}

TI_NAMESPACE_END
