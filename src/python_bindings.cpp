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

  py::class_<Expr>(m, "Expr").def("serialize", &Expr::serialize);
  // .def("initialize", &Texture::initialize);

  m.def("make_constant_expr", Expr::make<ConstExpression, int>);
  m.def("make_constant_expr", Expr::make<ConstExpression, float32>);
  m.def("make_constant_expr", Expr::make<ConstExpression, float64>);

  auto &&bin = py::enum_<BinaryOpType>(m, "BinaryOpType", py::arithmetic());
  for (int t = 0; t <= (int)BinaryOpType::undefined; t++)
    bin.value(binary_op_type_name(BinaryOpType(t)).c_str(), BinaryOpType(t));
  bin.export_values();
  m.def("make_binary_op_expr",
        Expr::make<BinaryOpExpression, const BinaryOpType &, const Expr &,
                   const Expr &>);

  auto &&unary = py::enum_<UnaryOpType>(m, "UnaryOpType", py::arithmetic());
  for (int t = 0; t <= (int)UnaryOpType::undefined; t++)
    unary.value(unary_op_type_name(UnaryOpType(t)).c_str(), UnaryOpType(t));
  unary.export_values();
  m.def("make_unary_op_expr",
        Expr::make<UnaryOpExpression, const UnaryOpType &, const Expr &>);
}

TLANG_NAMESPACE_END
