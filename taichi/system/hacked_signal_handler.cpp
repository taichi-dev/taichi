#include <csignal>

#include "taichi/python/export.h"
#include "taichi/common/logging.h"
#include "taichi/system/traceback.h"

namespace taichi {

namespace {

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

void signal_handler(int signo) {
  // It seems that there's no way to pass exception to Python in signal
  // handlers?
  // @archibate found that in fact there are such solution:
  // https://docs.python.org/3/library/faulthandler.html#module-faulthandler
  auto sig_name = signal_name(signo);
  logger.error(fmt::format("Received signal {} ({})", signo, sig_name), false);
  exit(-1);
  TI_UNREACHABLE;
}

class HackedSignalRegister {
 public:
  explicit HackedSignalRegister() {
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

#define TI_REGISTER_SIGNAL_HANDLER(name, handler)                   \
  {                                                                 \
    if (std::signal(name, handler) == SIG_ERR)                      \
      std::printf("Cannot register signal handler for" #name "\n"); \
  }

    TI_REGISTER_SIGNAL_HANDLER(SIGSEGV, signal_handler);
    TI_REGISTER_SIGNAL_HANDLER(SIGABRT, signal_handler);
#if !defined(_WIN64)
    TI_REGISTER_SIGNAL_HANDLER(SIGBUS, signal_handler);
#endif
    TI_REGISTER_SIGNAL_HANDLER(SIGFPE, signal_handler);

#undef TI_REGISTER_SIGNAL_HANDLER

    logger.set_print_stacktrace_func(print_traceback);
    TI_INFO("HackedSignalRegister initialized");
  }
};

HackedSignalRegister _;

}  // namespace
}  // namespace taichi
