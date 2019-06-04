#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <taichi/python/export.h>
#include <taichi/common/interface.h>
#include "tlang.h"

TLANG_NAMESPACE_BEGIN

Expr expr_index(const Expr &expr, const Expr &index) {
  return expr[index];
}

void expr_assign(const Expr &lhs, const Expr &rhs) {
  TC_ASSERT(lhs->is_lvalue());
  current_ast_builder().insert(
      std::make_unique<FrontendAssignStmt>(lhs, load_if_ptr(rhs)));
}

PYBIND11_MODULE(taichi_lang, m) {
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def("place", (SNode & (SNode::*)(Expr &))(&SNode::place));
  py::class_<Program>(m, "Program").def(py::init<>());
  py::class_<Program::Kernel>(m, "Kernel")
      .def("__call__", &Program::Kernel::operator());
  py::class_<Expr>(m, "Expr").def("serialize", &Expr::serialize);
  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("push_back", &ExprGroup::push_back)
      .def("serialize", &ExprGroup::serialize);
  py::class_<Stmt>(m, "Stmt");
  py::class_<Program::KernelProxy>(m, "KernelProxy")
      .def("define", &Program::KernelProxy::def,
           py::return_value_policy::reference);

  m.def("layout", layout);

  m.def("get_root", [&]() -> SNode * { return &root; },
        py::return_value_policy::reference);

  m.def("expr_add", expr_add);
  m.def("expr_sub", expr_sub);
  m.def("expr_mul", expr_mul);
  m.def("expr_div", expr_div);

  m.def("expr_cmp_le", expr_cmp_le);
  m.def("expr_cmp_lt", expr_cmp_lt);
  m.def("expr_cmp_ge", expr_cmp_ge);
  m.def("expr_cmp_gt", expr_cmp_gt);
  m.def("expr_cmp_ne", expr_cmp_ne);
  m.def("expr_cmp_eq", expr_cmp_eq);

  m.def("expr_index", expr_index);

  m.def("expr_var", [](const Expr &e) { return Var(e); });
  m.def("expr_assign", expr_assign);

  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>);
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>);
  m.def("make_frontend_assign_stmt",
        Stmt::make<FrontendAssignStmt, const Expr &, const Expr &>);

  m.def("make_id_expr", Expr::make<IdExpression, std::string>);
  m.def("make_constant_expr", Expr::make<ConstExpression, int>);
  m.def("make_constant_expr", Expr::make<ConstExpression, float32>);
  m.def("make_constant_expr", Expr::make<ConstExpression, float64>);

  m.def("make_global_ptr_expr",
        Expr::make<GlobalPtrExpression, const Expr &, const ExprGroup &>);

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

  auto &&data_type = py::enum_<DataType>(m, "DataType", py::arithmetic());
  for (int t = 0; t <= (int)DataType::unknown; t++)
    data_type.value(data_type_name(DataType(t)).c_str(), DataType(t));
  data_type.export_values();

  m.def("global_new", static_cast<Expr (*)(Expr, DataType)>(global_new));

  m.def("create_kernel", [&](std::string name) -> Program::KernelProxy {
    return get_current_program().kernel(name);
  });
}

TLANG_NAMESPACE_END
