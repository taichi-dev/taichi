#include <pybind11/pybind11.h>
#include <taichi/python/export.h>
#include <taichi/common/interface.h>

TC_NAMESPACE_BEGIN

void lang() {
  TC_TAG;
}

PYBIND11_MODULE(taichi_lang, m) {
  m.def("lang", &lang, "lang func");
}

TC_NAMESPACE_END
