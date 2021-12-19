#include <csignal>

#include "pybind11/pybind11.h"

namespace taichi {

namespace {

// Translates and propagates a CPP exception to Python.
// TODO(#2198): How can we glob all these global variables used as initializaers
// into a single function?
class ExceptionTranslationImpl {
 public:
  explicit ExceptionTranslationImpl() {
    pybind11::register_exception_translator([](std::exception_ptr p) {
      try {
        if (p)
          std::rethrow_exception(p);
      } catch (const std::string &e) {
        PyErr_SetString(PyExc_RuntimeError, e.c_str());
      }
    });
  }
};

ExceptionTranslationImpl _;

}  // namespace
}  // namespace taichi
