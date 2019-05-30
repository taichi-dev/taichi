#include <pybind11/pybind11.h>
#include <taichi/python/export.h>

TC_NAMESPACE_BEGIN

void export_tlang() {
  TC_TAG;
}

inline class Injector {
public:
  Injector() {
    extra_exports.push_back(export_tlang);
  }
} injector;

TC_NAMESPACE_END
