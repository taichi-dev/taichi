// Bindings for the python frontend

#include "tlang.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <taichi/common/interface.h>
#include <taichi/python/export.h>

TLANG_NAMESPACE_BEGIN

std::string compiled_lib_dir;

Expr expr_index(const Expr &expr, const Expr &index) { return expr[index]; }

void expr_assign(const Expr &lhs_, const Expr &rhs, std::string tb) {
  auto lhs = ptr_if_global(lhs_);
  TC_ASSERT(lhs->is_lvalue());
  auto stmt = std::make_unique<FrontendAssignStmt>(lhs, load_if_ptr(rhs));
  stmt->set_tb(tb);
  current_ast_builder().insert(std::move(stmt));
}

std::vector<std::unique_ptr<IRBuilder::ScopeGuard>> scope_stack;

void compile_runtimes();
std::string libdevice_path();

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
void export_lang(py::module &m) {
  using namespace taichi::Tlang;

  py::enum_<Arch>(m, "Arch", py::arithmetic())
      .value("gpu", Arch::gpu)
      .value("x86_64", Arch::x86_64)
      .export_values();

  py::class_<CompileConfig>(m, "CompileConfig")
      .def(py::init<>())
      .def_readwrite("arch", &CompileConfig::arch)
      .def_readwrite("print_ir", &CompileConfig::print_ir)
      .def_readwrite("use_llvm", &CompileConfig::use_llvm)
      .def_readwrite("print_struct_llvm_ir",
                     &CompileConfig::print_struct_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir",
                     &CompileConfig::print_kernel_llvm_ir)
      .def_readwrite("simplify_before_lower_access",
                     &CompileConfig::simplify_before_lower_access)
      .def_readwrite("simplify_after_lower_access",
                     &CompileConfig::simplify_after_lower_access)
      .def_readwrite("lower_access", &CompileConfig::lower_access)
      .def_readwrite("default_gpu_block_dim",
                     &CompileConfig::default_gpu_block_dim)
      .def_readwrite("verbose_kernel_launches",
                     &CompileConfig::verbose_kernel_launches)
      .def_readwrite("enable_profiler", &CompileConfig::enable_profiler)
      .def_readwrite("gradient_dt", &CompileConfig::gradient_dt);

  m.def("reset_default_compile_config",
        [&]() { default_compile_config = CompileConfig(); });

  m.def("default_compile_config",
        [&]() -> CompileConfig & { return default_compile_config; },
        py::return_value_policy::reference);

  py::class_<Program>(m, "Program")
      .def(py::init<>())
      .def_readonly("config", &Program::config)
      .def("clear_all_gradients", &Program::clear_all_gradients)
      .def("profiler_print", &Program::profiler_print)
      .def("profiler_print", &Program::profiler_clear)
      .def("finalize", &Program::finalize)
      .def("get_snode_writer", &Program::get_snode_writer)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("synchronize", &Program::synchronize);

  m.def("get_current_program", get_current_program,
        py::return_value_policy::reference);

  m.def("current_compile_config",
        [&]() -> CompileConfig & { return get_current_program().config; },
        py::return_value_policy::reference);

  py::class_<Index>(m, "Index").def(py::init<int>());
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def("clear_data", &SNode::clear_data)
      .def("clear_data_and_deactivate", &SNode::clear_data_and_deactivate)
      .def_readwrite("parent", &SNode::parent)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Index> &,
                               const std::vector<int> &))(&SNode::dense),
           py::return_value_policy::reference)
      .def("dynamic", &SNode::dynamic_chunked,
           py::return_value_policy::reference)
      .def("pointer", &SNode::pointer, py::return_value_policy::reference)
      .def("bitmasked", &SNode::bitmasked)
      .def("place", (SNode & (SNode::*)(Expr &))(&SNode::place),
           py::return_value_policy::reference)
      .def("data_type", [](SNode *snode) { return snode->dt; })
      .def("lazy_grad", &SNode::lazy_grad)
      .def("read_int", &SNode::read_int)
      .def("read_float", &SNode::read_float)
      .def("has_grad", &SNode::has_grad)
      .def("write_int", &SNode::write_int)
      .def("write_float", &SNode::write_float)
      .def("num_active_indices",
           [](SNode *snode) { return snode->num_active_indices; });

  py::class_<Kernel>(m, "Kernel")
      .def("set_arg_int", &Kernel::set_arg_int)
      .def("set_extra_arg_int", &Kernel::set_extra_arg_int)
      .def("set_arg_float", &Kernel::set_arg_float)
      .def("set_arg_nparray", &Kernel::set_arg_nparray)
      .def("__call__", &Kernel::operator());

  py::class_<Expr> expr(m, "Expr");
  expr.def("serialize", &Expr::serialize)
      .def("snode", &Expr::snode, py::return_value_policy::reference)
      .def("is_global_var",
           [](Expr *expr) { return expr->is<GlobalVariableExpression>(); })
      .def("set_tb", &Expr::set_tb)
      .def("set_is_primal",
           [&](Expr *expr, bool v) {
             expr->cast<GlobalVariableExpression>()->is_primal = v;
           })
      .def("set_grad", &Expr::set_grad)
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; });

  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("push_back", &ExprGroup::push_back)
      .def("serialize", &ExprGroup::serialize);
  py::class_<Stmt>(m, "Stmt");
  py::class_<Program::KernelProxy>(m, "KernelProxy")
      .def("define",
           [](Program::KernelProxy *ker,
              const std::function<void()> &func) -> Kernel & {
             py::gil_scoped_release release;
             return ker->def(func);
           },
           py::return_value_policy::reference);

  m.def("insert_append", [](SNode *snode, const ExprGroup &indices,
                            const Expr &val) { Append(snode, indices, val); });

  m.def("insert_len", [](SNode *snode, const ExprGroup &indices) {
    return Probe(snode, indices);
  });

  m.def("begin_frontend_while", [&](const Expr &cond) {
    auto stmt_unique = std::make_unique<FrontendWhileStmt>(cond);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    scope_stack.push_back(current_ast_builder().create_scope(stmt->body));
  });

  m.def("begin_frontend_range_for",
        [&](const Expr &i, const Expr &s, const Expr &e) {
          auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e);
          auto stmt = stmt_unique.get();
          current_ast_builder().insert(std::move(stmt_unique));
          scope_stack.push_back(current_ast_builder().create_scope(stmt->body));
        });

  m.def("begin_frontend_struct_for",
        [&](const ExprGroup &indices, const Expr &global) {
          auto stmt_unique = std::make_unique<FrontendForStmt>(indices, global);
          auto stmt = stmt_unique.get();
          current_ast_builder().insert(std::move(stmt_unique));
          scope_stack.push_back(current_ast_builder().create_scope(stmt->body));
        });

  m.def("end_frontend_range_for", [&]() { scope_stack.pop_back(); });
  m.def("pop_scope", [&]() { scope_stack.pop_back(); });

  m.def("begin_frontend_if", [&](const Expr &cond) {
    auto stmt_tmp = std::make_unique<FrontendIfStmt>(cond);
    current_ast_builder().insert(std::move(stmt_tmp));
  });

  m.def("begin_frontend_if_true", [&]() {
    auto if_stmt = current_ast_builder().get_last_stmt()->as<FrontendIfStmt>();
    scope_stack.push_back(
        current_ast_builder().create_scope(if_stmt->true_statements));
  });

  m.def("begin_frontend_if_false", [&]() {
    auto if_stmt = current_ast_builder().get_last_stmt()->as<FrontendIfStmt>();
    scope_stack.push_back(
        current_ast_builder().create_scope(if_stmt->false_statements));
  });

  m.def("layout", layout);

  m.def("get_root", [&]() -> SNode * { return &root; },
        py::return_value_policy::reference);

  m.def("value_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(cast));

  m.def("expr_atomic_add", [&](const Expr &a, const Expr &b) {
    current_ast_builder().insert(Stmt::make<FrontendAtomicStmt>(
        AtomicOpType::add, ptr_if_global(a), load_if_ptr(b)));
  });
  m.def("expr_add", expr_add);
  m.def("expr_sub", expr_sub);
  m.def("expr_mul", expr_mul);
  m.def("expr_div", expr_div);
  m.def("expr_mod", expr_mod);
  m.def("expr_max", expr_max);
  m.def("expr_min", expr_min);

  m.def("expr_bit_and", expr_bit_and);
  m.def("expr_bit_or", expr_bit_or);
  m.def("expr_bit_xor", expr_bit_xor);
  m.def("expr_bit_not", [](const Expr &expr) {
    return Expr::make<UnaryOpExpression>(UnaryOpType::logic_not, expr);
  });

  m.def("expr_cmp_le", expr_cmp_le);
  m.def("expr_cmp_lt", expr_cmp_lt);
  m.def("expr_cmp_ge", expr_cmp_ge);
  m.def("expr_cmp_gt", expr_cmp_gt);
  m.def("expr_cmp_ne", expr_cmp_ne);
  m.def("expr_cmp_eq", expr_cmp_eq);

  m.def("expr_index", expr_index);

#define DEFINE_EXPRESSION_OP_UNARY(x) m.def("expr_" #x, expr_##x);

  m.def("expr_neg", [&](const Expr &e) { return -e; });
  DEFINE_EXPRESSION_OP_UNARY(sqrt)
  DEFINE_EXPRESSION_OP_UNARY(floor)
  DEFINE_EXPRESSION_OP_UNARY(abs)
  DEFINE_EXPRESSION_OP_UNARY(sin)
  DEFINE_EXPRESSION_OP_UNARY(asin)
  DEFINE_EXPRESSION_OP_UNARY(cos)
  DEFINE_EXPRESSION_OP_UNARY(acos)
  DEFINE_EXPRESSION_OP_UNARY(tan)
  DEFINE_EXPRESSION_OP_UNARY(tanh)
  DEFINE_EXPRESSION_OP_UNARY(inv)
  DEFINE_EXPRESSION_OP_UNARY(rcp)
  DEFINE_EXPRESSION_OP_UNARY(rsqrt)
  DEFINE_EXPRESSION_OP_UNARY(exp)
  DEFINE_EXPRESSION_OP_UNARY(log)

  m.def("expr_var", [](const Expr &e) { return Var(e); });
  m.def("expr_alloca", []() {
    auto var = Expr(std::make_shared<IdExpression>());
    current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
        std::static_pointer_cast<IdExpression>(var.expr)->id,
        DataType::unknown));
    return var;
  });
  m.def("expr_assign", expr_assign);

  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>);
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>);
  m.def("make_frontend_assign_stmt",
        Stmt::make<FrontendAssignStmt, const Expr &, const Expr &>);

  m.def("make_arg_load_expr", Expr::make<ArgLoadExpression, int>);

  m.def("make_external_tensor_expr",
        Expr::make<ExternalTensorExpression, const DataType &, int, int>);

  m.def("make_id_expr", Expr::make<IdExpression, std::string>);

  m.def("make_rand_expr", Expr::make<RandExpression, const DataType &>);

  m.def("make_const_expr_i32", Expr::make<ConstExpression, int32>);
  m.def("make_const_expr_i64", Expr::make<ConstExpression, int64>);
  m.def("make_const_expr_f32", Expr::make<ConstExpression, float32>);
  m.def("make_const_expr_f64", Expr::make<ConstExpression, float64>);

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
  m.def("set_global_grad", [&](const Expr &expr) {
    TC_ASSERT(expr.is<GlobalVariableExpression>());
    expr.cast<GlobalVariableExpression>()->is_primal = false;
  });
  m.def("data_type_name", data_type_name);
  m.def("data_type_short_name", data_type_short_name);

  m.def("subscript", [](const Expr &expr, const ExprGroup &expr_group) {
    return expr[expr_group];
  });

  m.def("create_kernel",
        [&](std::string name, bool grad) -> Program::KernelProxy {
          return get_current_program().kernel(name, grad);
        });

  m.def("print_", Print_);

  m.def("decl_arg", [&](DataType dt, bool is_nparray) {
    return get_current_program().get_current_kernel().insert_arg(dt,
                                                                 is_nparray);
  });

  m.def("test_throw", [] {
    try {
      throw IRModified();
    } catch (IRModified) {
      TC_INFO("caught");
    }
  });
  // Schedules
  m.def("parallelize", Parallelize);
  m.def("vectorize", Vectorize);
  m.def("block_dim", BlockDim);
  m.def("cache", Cache);
  m.def("stop_grad",
        [](SNode *snode) { current_ast_builder().stop_gradient(snode); });

  m.def("test_throw", [] { throw IRModified(); });
  m.def("needs_grad", needs_grad);

  m.def("compile_runtimes", compile_runtimes);
  m.def("libdevice_path", libdevice_path);

  m.def("set_lib_dir", [&](const std::string &dir) { compiled_lib_dir = dir; });

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
  m.def("test_printf", [] {printf("test_printf\n");});
  m.def("test_logging", [] {TC_INFO("test_logging\n");});
  m.def("trigger_crash", [] {*(int *)(1)=0;});
}

TC_NAMESPACE_END
