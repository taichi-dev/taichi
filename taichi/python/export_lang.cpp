// Bindings for the python frontend

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"

#include "taichi/ir/frontend.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/extension.h"
#include "taichi/common/interface.h"
#include "taichi/python/export.h"
#include "taichi/gui/gui.h"
#include "taichi/math/svd.h"
#include "taichi/util/statistics.h"
#include "taichi/util/action_recorder.h"

#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/cuda_context.h"
#endif

TI_NAMESPACE_BEGIN

bool test_threading();

TI_NAMESPACE_END

TLANG_NAMESPACE_BEGIN

std::string compiled_lib_dir;
std::string runtime_tmp_dir;

Expr expr_index(const Expr &expr, const Expr &index) {
  return expr[index];
}

void expr_assign(const Expr &lhs_, const Expr &rhs, std::string tb) {
  auto lhs = ptr_if_global(lhs_);
  TI_ASSERT(lhs->is_lvalue());
  auto stmt = std::make_unique<FrontendAssignStmt>(lhs, load_if_ptr(rhs));
  stmt->set_tb(tb);
  current_ast_builder().insert(std::move(stmt));
}

std::vector<std::unique_ptr<IRBuilder::ScopeGuard>> scope_stack;

void compile_runtimes();
std::string libdevice_path();
std::string get_runtime_dir();

TLANG_NAMESPACE_END

TI_NAMESPACE_BEGIN
void export_lang(py::module &m) {
  using namespace taichi::lang;

  py::enum_<Arch>(m, "Arch", py::arithmetic())
#define PER_ARCH(x) .value(#x, Arch::x)
#include "taichi/inc/archs.inc.h"
#undef PER_ARCH
      .export_values();

  m.def("arch_name", arch_name);
  m.def("arch_from_name", arch_from_name);

  py::enum_<SNodeType>(m, "SNodeType", py::arithmetic())
#define PER_SNODE(x) .value(#x, SNodeType::x)
#include "taichi/inc/snodes.inc.h"
#undef PER_SNODE
      .export_values();

  py::enum_<Extension>(m, "Extension", py::arithmetic())
#define PER_EXTENSION(x) .value(#x, Extension::x)
#include "taichi/inc/extensions.inc.h"
#undef PER_EXTENSION
      .export_values();

  py::class_<CompileConfig>(m, "CompileConfig")
      .def(py::init<>())
      .def_readwrite("arch", &CompileConfig::arch)
      .def_readwrite("print_ir", &CompileConfig::print_ir)
      .def_readwrite("debug", &CompileConfig::debug)
      .def_readwrite("check_out_of_bound", &CompileConfig::check_out_of_bound)
      .def_readwrite("print_accessor_ir", &CompileConfig::print_accessor_ir)
      .def_readwrite("print_evaluator_ir", &CompileConfig::print_evaluator_ir)
      .def_readwrite("use_llvm", &CompileConfig::use_llvm)
      .def_readwrite("print_benchmark_stat",
                     &CompileConfig::print_benchmark_stat)
      .def_readwrite("print_struct_llvm_ir",
                     &CompileConfig::print_struct_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir",
                     &CompileConfig::print_kernel_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir_optimized",
                     &CompileConfig::print_kernel_llvm_ir_optimized)
      .def_readwrite("print_kernel_nvptx", &CompileConfig::print_kernel_nvptx)
      .def_readwrite("simplify_before_lower_access",
                     &CompileConfig::simplify_before_lower_access)
      .def_readwrite("simplify_after_lower_access",
                     &CompileConfig::simplify_after_lower_access)
      .def_readwrite("lower_access", &CompileConfig::lower_access)
      .def_readwrite("default_cpu_block_dim",
                     &CompileConfig::default_cpu_block_dim)
      .def_readwrite("default_gpu_block_dim",
                     &CompileConfig::default_gpu_block_dim)
      .def_readwrite("saturating_grid_dim", &CompileConfig::saturating_grid_dim)
      .def_readwrite("max_block_dim", &CompileConfig::max_block_dim)
      .def_readwrite("verbose_kernel_launches",
                     &CompileConfig::verbose_kernel_launches)
      .def_readwrite("verbose", &CompileConfig::verbose)
      .def_readwrite("demote_dense_struct_fors",
                     &CompileConfig::demote_dense_struct_fors)
      .def_readwrite("use_unified_memory", &CompileConfig::use_unified_memory)
      .def_readwrite("kernel_profiler", &CompileConfig::kernel_profiler)
      .def_readwrite("default_fp", &CompileConfig::default_fp)
      .def_readwrite("default_ip", &CompileConfig::default_ip)
      .def_readwrite("device_memory_GB", &CompileConfig::device_memory_GB)
      .def_readwrite("device_memory_fraction",
                     &CompileConfig::device_memory_fraction)
      .def_readwrite("fast_math", &CompileConfig::fast_math)
      .def_readwrite("advanced_optimization",
                     &CompileConfig::advanced_optimization)
      .def_readwrite("ad_stack_size", &CompileConfig::ad_stack_size)
      .def_readwrite("async_mode", &CompileConfig::async_mode)
      .def_readwrite("flatten_if", &CompileConfig::flatten_if)
      .def_readwrite("make_thread_local", &CompileConfig::make_thread_local)
      .def_readwrite("cc_compile_cmd", &CompileConfig::cc_compile_cmd)
      .def_readwrite("cc_link_cmd", &CompileConfig::cc_link_cmd);

  m.def("reset_default_compile_config",
        [&]() { default_compile_config = CompileConfig(); });

  m.def("default_compile_config",
        [&]() -> CompileConfig & { return default_compile_config; },
        py::return_value_policy::reference);

  py::class_<Program>(m, "Program")
      .def(py::init<>())
      .def_readonly("config", &Program::config)
      .def("kernel_profiler_print", &Program::kernel_profiler_print)
      .def("kernel_profiler_clear", &Program::kernel_profiler_clear)
      .def("print_memory_profiler_info", &Program::print_memory_profiler_info)
      .def("finalize", &Program::finalize)
      .def("get_root",
           [&](Program *program) -> SNode * {
             return program->snode_root.get();
           },
           py::return_value_policy::reference)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("print_snode_tree", &Program::print_snode_tree)
      .def("get_snode_num_dynamically_allocated",
           &Program::get_snode_num_dynamically_allocated)
      .def("synchronize", &Program::synchronize);

  m.def("get_current_program", get_current_program,
        py::return_value_policy::reference);

  m.def("current_compile_config",
        [&]() -> CompileConfig & { return get_current_program().config; },
        py::return_value_policy::reference);

  py::class_<Index>(m, "Index").def(py::init<int>());
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def_readwrite("parent", &SNode::parent)
      .def_readonly("type", &SNode::type)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Index> &,
                               const std::vector<int> &))(&SNode::dense),
           py::return_value_policy::reference)
      .def("pointer",
           (SNode & (SNode::*)(const std::vector<Index> &,
                               const std::vector<int> &))(&SNode::pointer),
           py::return_value_policy::reference)
      .def("hash",
           (SNode & (SNode::*)(const std::vector<Index> &,
                               const std::vector<int> &))(&SNode::hash),
           py::return_value_policy::reference)
      .def("dynamic", &SNode::dynamic, py::return_value_policy::reference)
      .def("bitmasked",
           (SNode & (SNode::*)(const std::vector<Index> &,
                               const std::vector<int> &))(&SNode::bitmasked),
           py::return_value_policy::reference)
      .def("place",
           (void (SNode::*)(Expr &, const std::vector<int> &))(&SNode::place),
           py::return_value_policy::reference)
      .def("data_type", [](SNode *snode) { return snode->dt; })
      .def("get_num_ch",
           [](SNode *snode) -> int { return (int)snode->ch.size(); })
      .def("get_ch",
           [](SNode *snode, int i) -> SNode * { return snode->ch[i].get(); },
           py::return_value_policy::reference)
      .def("lazy_grad", &SNode::lazy_grad)
      .def("read_int", &SNode::read_int)
      .def("read_uint", &SNode::read_uint)
      .def("read_float", &SNode::read_float)
      .def("has_grad", &SNode::has_grad)
      .def("is_primal", &SNode::is_primal)
      .def("is_place", &SNode::is_place)
      .def("get_expr", &SNode::get_expr, py::return_value_policy::reference)
      .def("write_int", &SNode::write_int)
      .def("write_float", &SNode::write_float)
      .def("get_shape_along_axis", &SNode::shape_along_axis)
      .def("get_physical_index_position",
           [](SNode *snode) {
             return std::vector<int>(
                 snode->physical_index_position,
                 snode->physical_index_position + taichi_max_num_indices);
           })
      .def("num_active_indices",
           [](SNode *snode) { return snode->num_active_indices; });

  py::class_<Kernel>(m, "Kernel")
      .def("get_ret_int", &Kernel::get_ret_int)
      .def("get_ret_float", &Kernel::get_ret_float)
      .def("make_launch_context", &Kernel::make_launch_context)
      .def("__call__",
           [](Kernel *kernel, Kernel::LaunchContextBuilder &launch_ctx) {
             py::gil_scoped_release release;
             kernel->operator()(launch_ctx);
           });

  py::class_<Kernel::LaunchContextBuilder>(m, "KernelLaunchContext")
      .def("set_arg_int", &Kernel::LaunchContextBuilder::set_arg_int)
      .def("set_arg_float", &Kernel::LaunchContextBuilder::set_arg_float)
      .def("set_arg_nparray", &Kernel::LaunchContextBuilder::set_arg_nparray)
      .def("set_extra_arg_int",
           &Kernel::LaunchContextBuilder::set_extra_arg_int);

  py::class_<Expr> expr(m, "Expr");
  expr.def("serialize", &Expr::serialize)
      .def("snode", &Expr::snode, py::return_value_policy::reference)
      .def("is_global_var",
           [](Expr *expr) { return expr->is<GlobalVariableExpression>(); })
      .def("is_external_var",
           [](Expr *expr) { return expr->is<ExternalTensorExpression>(); })
      .def("set_tb", &Expr::set_tb)
      .def("set_is_primal",
           [&](Expr *expr, bool v) {
             expr->cast<GlobalVariableExpression>()->is_primal = v;
           })
      .def("set_grad", &Expr::set_grad)
      .def("set_attribute", &Expr::set_attribute)
      .def("get_attribute", &Expr::get_attribute)
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; });

  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("size", [](ExprGroup *eg) { return eg->exprs.size(); })
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

  m.def("insert_deactivate", [](SNode *snode, const ExprGroup &indices) {
    return Deactivate(snode, indices);
  });

  m.def("insert_activate", [](SNode *snode, const ExprGroup &indices) {
    return Activate(snode, indices);
  });

  m.def("insert_append",
        [](SNode *snode, const ExprGroup &indices, const Expr &val) {
          return Append(snode, indices, val);
        });

  m.def("insert_external_func_call",
        [](std::size_t func_addr, std::string source, const ExprGroup &args,
           const ExprGroup &outputs) {
          auto expr = Expr::make<ExternalFuncCallExpression>(
              (void *)func_addr, source, args.exprs, outputs.exprs);

          current_ast_builder().insert(Stmt::make<FrontendEvalStmt>(expr));
        });

  m.def("insert_is_active", [](SNode *snode, const ExprGroup &indices) {
    return is_active(snode, indices);
  });

  m.def("insert_len", [](SNode *snode, const ExprGroup &indices) {
    return Length(snode, indices);
  });

  m.def("create_assert_stmt", [&](const Expr &cond, const std::string &msg) {
    auto stmt_unique = std::make_unique<FrontendAssertStmt>(msg, cond);
    current_ast_builder().insert(std::move(stmt_unique));
  });

  m.def("create_internal_func_stmt", [&](const std::string &msg) {
    current_ast_builder().insert(std::make_unique<InternalFuncStmt>(msg));
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

  m.def("insert_break_stmt", [&]() {
    current_ast_builder().insert(Stmt::make<FrontendBreakStmt>());
  });

  m.def("create_kernel_return", [&](const Expr &value) {
    current_ast_builder().insert(Stmt::make<FrontendKernelReturnStmt>(value));
  });

  m.def("insert_continue_stmt", [&]() {
    current_ast_builder().insert(Stmt::make<FrontendContinueStmt>());
  });

  m.def("begin_func", [&](const std::string &funcid) {
    auto stmt_unique = std::make_unique<FrontendFuncDefStmt>(funcid);
    auto stmt = stmt_unique.get();
    current_ast_builder().insert(std::move(stmt_unique));
    scope_stack.push_back(current_ast_builder().create_scope(stmt->body));
  });

  m.def("end_func", [&](const std::string &funcid) { scope_stack.pop_back(); });

  m.def("func_call", [&](const std::string &funcid) {
    auto func = Stmt::make<FuncCallStmt>(
        funcid);  // TODO: use FuncCallExpr with return values & args
    current_ast_builder().insert(std::move(func));
  });

  m.def("layout", layout);

  m.def("value_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(cast));
  m.def("bits_cast",
        static_cast<Expr (*)(const Expr &expr, DataType)>(bit_cast));

  m.def("expr_atomic_add", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::add, ptr_if_global(a),
                                          load_if_ptr(b));
  });

  m.def("expr_atomic_sub", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::sub, ptr_if_global(a),
                                          load_if_ptr(b));
  });

  m.def("expr_atomic_min", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::min, ptr_if_global(a),
                                          load_if_ptr(b));
  });

  m.def("expr_atomic_max", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::max, ptr_if_global(a),
                                          load_if_ptr(b));
  });

  m.def("expr_atomic_bit_and", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_and,
                                          ptr_if_global(a), load_if_ptr(b));
  });

  m.def("expr_atomic_bit_or", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_or,
                                          ptr_if_global(a), load_if_ptr(b));
  });

  m.def("expr_atomic_bit_xor", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_xor,
                                          ptr_if_global(a), load_if_ptr(b));
  });

  m.def("expr_add", expr_add);
  m.def("expr_sub", expr_sub);
  m.def("expr_mul", expr_mul);
  m.def("expr_div", expr_div);
  m.def("expr_truediv", expr_truediv);
  m.def("expr_floordiv", expr_floordiv);
  m.def("expr_mod", expr_mod);
  m.def("expr_max", expr_max);
  m.def("expr_min", expr_min);
  m.def("expr_atan2", expr_atan2);
  m.def("expr_pow", expr_pow);

  m.def("expr_bit_and", expr_bit_and);
  m.def("expr_bit_or", expr_bit_or);
  m.def("expr_bit_xor", expr_bit_xor);
  m.def("expr_bit_not", expr_bit_not);
  m.def("expr_logic_not", expr_logic_not);

  m.def("expr_cmp_le", expr_cmp_le);
  m.def("expr_cmp_lt", expr_cmp_lt);
  m.def("expr_cmp_ge", expr_cmp_ge);
  m.def("expr_cmp_gt", expr_cmp_gt);
  m.def("expr_cmp_ne", expr_cmp_ne);
  m.def("expr_cmp_eq", expr_cmp_eq);

  m.def("expr_index", expr_index);

  m.def("expr_assume_in_range", AssumeInRange);

  m.def("expr_select", expr_select);

#define DEFINE_EXPRESSION_OP_UNARY(x) m.def("expr_" #x, expr_##x);

  m.def("expr_neg", [&](const Expr &e) { return -e; });
  DEFINE_EXPRESSION_OP_UNARY(sqrt)
  DEFINE_EXPRESSION_OP_UNARY(floor)
  DEFINE_EXPRESSION_OP_UNARY(ceil)
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

  m.def("is_integral", is_integral);
  m.def("is_signed", is_signed);
  m.def("is_unsigned", is_unsigned);

  m.def("global_new", static_cast<Expr (*)(Expr, DataType)>(global_new));
  m.def("set_global_grad", [&](const Expr &expr) {
    TI_ASSERT(expr.is<GlobalVariableExpression>());
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

  m.def("create_print",
        [&](std::vector<std::variant<Expr, std::string>> contents) {
          current_ast_builder().insert(
              std::make_unique<FrontendPrintStmt>(contents));
        });

  m.def("decl_arg", [&](DataType dt, bool is_nparray) {
    return get_current_program().get_current_kernel().insert_arg(dt,
                                                                 is_nparray);
  });

  m.def("decl_ret", [&](DataType dt) {
    return get_current_program().get_current_kernel().insert_ret(dt);
  });

  m.def("test_throw", [] {
    try {
      throw IRModified();
    } catch (IRModified) {
      TI_INFO("caught");
    }
  });
  // Schedules
  m.def("parallelize", Parallelize);
  m.def("vectorize", Vectorize);
  m.def("block_dim", BlockDim);
  m.def("cache", Cache);
  m.def("no_activate", [](SNode *snode) {
    get_current_program().get_current_kernel().no_activate.push_back(snode);
  });
  m.def("stop_grad",
        [](SNode *snode) { current_ast_builder().stop_gradient(snode); });

  m.def("test_throw", [] { throw IRModified(); });
  m.def("needs_grad", needs_grad);

  m.def("compile_runtimes", compile_runtimes);
  m.def("libdevice_path", libdevice_path);

  m.def("host_arch", host_arch);

  m.def("set_lib_dir", [&](const std::string &dir) { compiled_lib_dir = dir; });
  m.def("set_tmp_dir", [&](const std::string &dir) { runtime_tmp_dir = dir; });
  m.def("get_runtime_dir", get_runtime_dir);

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
  m.def("get_llvm_version_string", get_llvm_version_string);
  m.def("test_printf", [] { printf("test_printf\n"); });
  m.def("test_logging", [] { TI_INFO("test_logging"); });
  m.def("trigger_crash", [] { *(int *)(1) = 0; });
  m.def("get_max_num_indices", [] { return taichi_max_num_indices; });
  m.def("get_max_num_args", [] { return taichi_max_num_args; });
  m.def("test_threading", test_threading);
  m.def("sifakis_svd_f32", sifakis_svd_export<float32, int32>);
  m.def("sifakis_svd_f64", sifakis_svd_export<float64, int64>);
  m.def("global_var_expr_from_snode", [](SNode *snode) {
    return Expr::make<GlobalVariableExpression>(snode);
  });
  m.def("is_extension_supported", is_extension_supported);

  m.def("print_stat", [] { stat.print(); });
  m.def("stat", [] {
    std::string result;
    stat.print(&result);
    return result;
  });

  m.def("record_action_hint", [](std::string content) {
    ActionRecorder::get_instance().record("hint",
                                          {ActionArg("content", content)});
  });

  m.def("start_recording", [](const std::string &fn) {
    ActionRecorder::get_instance().start_recording(fn);
  });

  m.def("stop_recording",
        []() { ActionRecorder::get_instance().stop_recording(); });

  // A temporary option which will be removed soon in the future
  m.def("toggle_advanced_optimization", [](bool option) {
    TI_WARN(
        "'ti.core.toggle_advance_optimization(False)' is deprecated."
        " Use 'ti.init(advanced_optimization=False)' instead");
    get_current_program().config.advanced_optimization = option;
  });

  m.def("query_int64", [](const std::string &key) {
    if (key == "cuda_compute_capability") {
#if defined(TI_WITH_CUDA)
      return CUDAContext::get_instance().get_compute_capability();
#else
      TI_NOT_IMPLEMENTED
#endif
    } else {
      TI_ERROR("Key {} not supported in query_int64", key);
    }
  });
}

TI_NAMESPACE_END
