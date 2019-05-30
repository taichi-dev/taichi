#include <pybind11/pybind11.h>
#include <taichi/python/export.h>
#include <taichi/common/interface.h>
#include "tlang.h"

TLANG_NAMESPACE_BEGIN

void lang() {
  TC_TAG;
}

PYBIND11_MODULE(taichi_lang, m) {
  m.def("lang", &lang, "lang func");
  py::class_<Expr>(m, "Expr");
  // .def("initialize", &Texture::initialize);
  m.def("make_constant_expr", Expr::make<ConstExpression, int>);
  m.def("make_constant_expr", Expr::make<ConstExpression, float32>);
  m.def("make_constant_expr", Expr::make<ConstExpression, float64>);

  auto &&bin = py::enum_<BinaryType>(m, "BinaryType", py::arithmetic());

  for (int t = 0; t <= (int)BinaryType::undefined; t++) {
    bin.value(binary_type_name(BinaryType(t)).c_str(), BinaryType(t));
  }

  bin.export_values();

  m.def("make_binary_op_expr",
        Expr::make<BinaryOpExpression, const BinaryType &, const Expr &, const Expr &>);
}

TLANG_NAMESPACE_END
