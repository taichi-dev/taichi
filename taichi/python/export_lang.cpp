// Bindings for the python frontend

#include <optional>
#include <string>
#include "taichi/ir/snode.h"

#if TI_WITH_LLVM
#include "llvm/Config/llvm-config.h"
#endif

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"

#include "taichi/ir/frontend.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/program/async_engine.h"
#include "taichi/program/ndarray.h"
#include "taichi/common/interface.h"
#include "taichi/python/export.h"
#include "taichi/gui/gui.h"
#include "taichi/math/svd.h"
#include "taichi/util/statistics.h"
#include "taichi/util/action_recorder.h"
#include "taichi/system/timeline.h"
#include "taichi/python/snode_registry.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/sparse_solver.h"
#include "taichi/ir/mesh.h"

#include "taichi/program/kernel_profiler.h"

#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/cuda_context.h"
#endif

TI_NAMESPACE_BEGIN
bool test_threading();

TI_NAMESPACE_END

TLANG_NAMESPACE_BEGIN
void async_print_sfg();

std::string async_dump_dot(std::optional<std::string> rankdir,
                           int embed_states_threshold);

Expr expr_index(const Expr &expr, const Expr &index) {
  return expr[index];
}

std::string libdevice_path();

TLANG_NAMESPACE_END

TI_NAMESPACE_BEGIN
void export_lang(py::module &m) {
  using namespace taichi::lang;

  py::register_exception<TaichiTypeError>(m, "TaichiTypeError",
                                          PyExc_TypeError);
  py::register_exception<TaichiSyntaxError>(m, "TaichiSyntaxError",
                                            PyExc_SyntaxError);
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

  // TODO(type): This should be removed
  py::class_<DataType>(m, "DataType")
      .def(py::init<Type *>())
      .def(py::self == py::self)
      .def("__hash__", &DataType::hash)
      .def("to_string", &DataType::to_string)
      .def("__str__", &DataType::to_string)
      .def(
          "get_ptr", [](DataType *dtype) -> Type * { return *dtype; },
          py::return_value_policy::reference)
      .def(py::pickle(
          [](const DataType &dt) {
            // Note: this only works for primitive types, which is fine for now.
            auto primitive =
                dynamic_cast<const PrimitiveType *>((const Type *)dt);
            TI_ASSERT(primitive);
            return py::make_tuple((std::size_t)primitive->type);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid state!");

            DataType dt =
                PrimitiveType::get((PrimitiveTypeID)(t[0].cast<std::size_t>()));

            return dt;
          }));

  py::class_<CompileConfig>(m, "CompileConfig")
      .def(py::init<>())
      .def_readwrite("arch", &CompileConfig::arch)
      .def_readwrite("opt_level", &CompileConfig::opt_level)
      .def_readwrite("packed", &CompileConfig::packed)
      .def_readwrite("print_ir", &CompileConfig::print_ir)
      .def_readwrite("print_preprocessed_ir",
                     &CompileConfig::print_preprocessed_ir)
      .def_readwrite("debug", &CompileConfig::debug)
      .def_readwrite("cfg_optimization", &CompileConfig::cfg_optimization)
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
      .def_readwrite("move_loop_invariant_outside_if",
                     &CompileConfig::move_loop_invariant_outside_if)
      .def_readwrite("default_cpu_block_dim",
                     &CompileConfig::default_cpu_block_dim)
      .def_readwrite("default_gpu_block_dim",
                     &CompileConfig::default_gpu_block_dim)
      .def_readwrite("gpu_max_reg", &CompileConfig::gpu_max_reg)
      .def_readwrite("saturating_grid_dim", &CompileConfig::saturating_grid_dim)
      .def_readwrite("max_block_dim", &CompileConfig::max_block_dim)
      .def_readwrite("cpu_max_num_threads", &CompileConfig::cpu_max_num_threads)
      .def_readwrite("random_seed", &CompileConfig::random_seed)
      .def_readwrite("verbose_kernel_launches",
                     &CompileConfig::verbose_kernel_launches)
      .def_readwrite("verbose", &CompileConfig::verbose)
      .def_readwrite("demote_dense_struct_fors",
                     &CompileConfig::demote_dense_struct_fors)
      .def_readwrite("kernel_profiler", &CompileConfig::kernel_profiler)
      .def_readwrite("timeline", &CompileConfig::timeline)
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
      .def_readwrite("dynamic_index", &CompileConfig::dynamic_index)
      .def_readwrite("flatten_if", &CompileConfig::flatten_if)
      .def_readwrite("make_thread_local", &CompileConfig::make_thread_local)
      .def_readwrite("make_block_local", &CompileConfig::make_block_local)
      .def_readwrite("detect_read_only", &CompileConfig::detect_read_only)
      .def_readwrite("ndarray_use_cached_allocator",
                     &CompileConfig::ndarray_use_cached_allocator)
      .def_readwrite("cc_compile_cmd", &CompileConfig::cc_compile_cmd)
      .def_readwrite("cc_link_cmd", &CompileConfig::cc_link_cmd)
      .def_readwrite("async_opt_passes", &CompileConfig::async_opt_passes)
      .def_readwrite("async_opt_fusion", &CompileConfig::async_opt_fusion)
      .def_readwrite("async_opt_fusion_max_iter",
                     &CompileConfig::async_opt_fusion_max_iter)
      .def_readwrite("async_opt_listgen", &CompileConfig::async_opt_listgen)
      .def_readwrite("async_opt_activation_demotion",
                     &CompileConfig::async_opt_activation_demotion)
      .def_readwrite("async_opt_dse", &CompileConfig::async_opt_dse)
      .def_readwrite("async_listgen_fast_filtering",
                     &CompileConfig::async_listgen_fast_filtering)
      .def_readwrite("async_opt_intermediate_file",
                     &CompileConfig::async_opt_intermediate_file)
      .def_readwrite("async_flush_every", &CompileConfig::async_flush_every)
      .def_readwrite("async_max_fuse_per_task",
                     &CompileConfig::async_max_fuse_per_task)
      .def_readwrite("quant_opt_store_fusion",
                     &CompileConfig::quant_opt_store_fusion)
      .def_readwrite("quant_opt_atomic_demotion",
                     &CompileConfig::quant_opt_atomic_demotion)
      .def_readwrite("allow_nv_shader_extension",
                     &CompileConfig::allow_nv_shader_extension)
      .def_readwrite("use_gles", &CompileConfig::use_gles)
      .def_readwrite("make_mesh_block_local",
                     &CompileConfig::make_mesh_block_local)
      .def_readwrite("mesh_localize_to_end_mapping",
                     &CompileConfig::mesh_localize_to_end_mapping)
      .def_readwrite("mesh_localize_from_end_mapping",
                     &CompileConfig::mesh_localize_from_end_mapping)
      .def_readwrite("optimize_mesh_reordered_mapping",
                     &CompileConfig::optimize_mesh_reordered_mapping)
      .def_readwrite("mesh_localize_all_attr_mappings",
                     &CompileConfig::mesh_localize_all_attr_mappings)
      .def_readwrite("demote_no_access_mesh_fors",
                     &CompileConfig::demote_no_access_mesh_fors)
      .def_readwrite("experimental_auto_mesh_local",
                     &CompileConfig::experimental_auto_mesh_local)
      .def_readwrite("auto_mesh_local_default_occupacy",
                     &CompileConfig::auto_mesh_local_default_occupacy);

  m.def("reset_default_compile_config",
        [&]() { default_compile_config = CompileConfig(); });

  m.def(
      "default_compile_config",
      [&]() -> CompileConfig & { return default_compile_config; },
      py::return_value_policy::reference);

  py::class_<Program::KernelProfilerQueryResult>(m, "KernelProfilerQueryResult")
      .def_readwrite("counter", &Program::KernelProfilerQueryResult::counter)
      .def_readwrite("min", &Program::KernelProfilerQueryResult::min)
      .def_readwrite("max", &Program::KernelProfilerQueryResult::max)
      .def_readwrite("avg", &Program::KernelProfilerQueryResult::avg);

  py::class_<KernelProfileTracedRecord>(m, "KernelProfileTracedRecord")
      .def_readwrite("register_per_thread",
                     &KernelProfileTracedRecord::register_per_thread)
      .def_readwrite("shared_mem_per_block",
                     &KernelProfileTracedRecord::shared_mem_per_block)
      .def_readwrite("grid_size", &KernelProfileTracedRecord::grid_size)
      .def_readwrite("block_size", &KernelProfileTracedRecord::block_size)
      .def_readwrite(
          "active_blocks_per_multiprocessor",
          &KernelProfileTracedRecord::active_blocks_per_multiprocessor)
      .def_readwrite("kernel_time",
                     &KernelProfileTracedRecord::kernel_elapsed_time_in_ms)
      .def_readwrite("base_time", &KernelProfileTracedRecord::time_since_base)
      .def_readwrite("name", &KernelProfileTracedRecord::name)
      .def_readwrite("metric_values",
                     &KernelProfileTracedRecord::metric_values);

  // Export ASTBuilder
  py::class_<ASTBuilder>(m, "ASTBuilder")
      .def("create_kernel_exprgroup_return",
           &ASTBuilder::create_kernel_exprgroup_return)
      .def("create_print", &ASTBuilder::create_print)
      .def("begin_func", &ASTBuilder::begin_func)
      .def("end_func", &ASTBuilder::end_func)
      .def("stop_grad", &ASTBuilder::stop_gradient)
      .def("begin_frontend_if", &ASTBuilder::begin_frontend_if)
      .def("begin_frontend_if_true", &ASTBuilder::begin_frontend_if_true)
      .def("pop_scope", &ASTBuilder::pop_scope)
      .def("begin_frontend_if_false", &ASTBuilder::begin_frontend_if_false)
      .def("insert_deactivate", Deactivate)
      .def("insert_activate", Activate)
      .def("insert_external_func_call", &ASTBuilder::insert_external_func_call)
      .def("expr_alloca", &ASTBuilder::expr_alloca)
      .def("expr_alloca_local_tensor", &ASTBuilder::expr_alloca_local_tensor)
      .def("create_assert_stmt", &ASTBuilder::create_assert_stmt)
      .def("expr_assign", &ASTBuilder::expr_assign)
      .def("begin_frontend_range_for", &ASTBuilder::begin_frontend_range_for)
      .def("end_frontend_range_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_struct_for", &ASTBuilder::begin_frontend_struct_for)
      .def("end_frontend_struct_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_mesh_for", &ASTBuilder::begin_frontend_mesh_for)
      .def("end_frontend_mesh_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_while", &ASTBuilder::begin_frontend_while)
      .def("insert_break_stmt", &ASTBuilder::insert_break_stmt)
      .def("insert_continue_stmt", &ASTBuilder::insert_continue_stmt)
      .def("insert_expr_stmt", &ASTBuilder::insert_expr_stmt)
      .def("insert_thread_idx_expr", &ASTBuilder::insert_thread_idx_expr)
      .def("insert_patch_idx_expr", &ASTBuilder::insert_patch_idx_expr)
      .def("sifakis_svd_f32", sifakis_svd_export<float32, int32>)
      .def("sifakis_svd_f64", sifakis_svd_export<float64, int64>)
      .def("expr_var", &ASTBuilder::make_var);

  py::class_<Program>(m, "Program")
      .def(py::init<>())
      .def_readonly("config", &Program::config)
      .def("sync_kernel_profiler",
           [](Program *program) { program->profiler->sync(); })
      .def("query_kernel_profile_info",
           [](Program *program, const std::string &name) {
             return program->query_kernel_profile_info(name);
           })
      .def("get_kernel_profiler_records",
           [](Program *program) {
             return program->profiler->get_traced_records();
           })
      .def(
          "get_kernel_profiler_device_name",
          [](Program *program) { return program->profiler->get_device_name(); })
      .def("reinit_kernel_profiler_with_metrics",
           [](Program *program, const std::vector<std::string> metrics) {
             return program->profiler->reinit_with_metrics(metrics);
           })
      .def("kernel_profiler_total_time",
           [](Program *program) { return program->profiler->get_total_time(); })
      .def("set_kernel_profiler_toolkit",
           [](Program *program, const std::string toolkit_name) {
             return program->profiler->set_profiler_toolkit(toolkit_name);
           })
      .def("clear_kernel_profile_info", &Program::clear_kernel_profile_info)
      .def("timeline_clear",
           [](Program *) { Timelines::get_instance().clear(); })
      .def("timeline_save",
           [](Program *, const std::string &fn) {
             Timelines::get_instance().save(fn);
           })
      .def("print_memory_profiler_info", &Program::print_memory_profiler_info)
      .def("finalize", &Program::finalize)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("visualize_layout", &Program::visualize_layout)
      .def("get_snode_num_dynamically_allocated",
           &Program::get_snode_num_dynamically_allocated)
      .def("benchmark_rebuild_graph",
           [](Program *program) {
             program->async_engine->sfg->benchmark_rebuild_graph();
           })
      .def("synchronize", &Program::synchronize)
      .def("async_flush", &Program::async_flush)
      .def("materialize_runtime", &Program::materialize_runtime)
      .def("make_aot_module_builder", &Program::make_aot_module_builder)
      .def("get_snode_tree_size", &Program::get_snode_tree_size)
      .def("get_snode_root", &Program::get_snode_root,
           py::return_value_policy::reference)
      .def("current_ast_builder", &Program::current_ast_builder,
           py::return_value_policy::reference)
      .def(
          "create_kernel",
          [](Program *program, const std::function<void(Kernel *)> &body,
             const std::string &name, bool grad) -> Kernel * {
            py::gil_scoped_release release;
            return &program->kernel(body, name, grad);
          },
          py::return_value_policy::reference)
      .def("create_function", &Program::create_function,
           py::return_value_policy::reference)
      .def("create_sparse_matrix_builder",
           [](Program *program, int n, int m, uint64 max_num_entries) {
             TI_ERROR_IF(!arch_is_cpu(program->config.arch),
                         "SparseMatrix only supports CPU for now.");
             return SparseMatrixBuilder(n, m, max_num_entries);
           })
      .def("create_sparse_matrix",
           [](Program *program, int n, int m) {
             TI_ERROR_IF(!arch_is_cpu(program->config.arch),
                         "SparseMatrix only supports CPU for now.");
             return SparseMatrix(n, m);
           })
      .def(
          "dump_dot",
          [](Program *program, std::optional<std::string> rankdir,
             int embed_states_threshold) {
            // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#allow-prohibiting-none-arguments
            return program->async_engine->sfg->dump_dot(rankdir,
                                                        embed_states_threshold);
          },
          py::arg("rankdir").none(true),
          py::arg("embed_states_threshold"))  // FIXME:
      .def("no_activate",
           [](Program *program, SNode *snode) {
             // TODO(#2193): Also apply to @ti.func?
             auto *kernel = dynamic_cast<Kernel *>(program->current_callable);
             TI_ASSERT(kernel);
             kernel->no_activate.push_back(snode);
           })
      .def("print_sfg",
           [](Program *program) { return program->async_engine->sfg->print(); })
      .def("decl_arg",
           [&](Program *program, const DataType &dt, bool is_array) {
             return program->current_callable->insert_arg(dt, is_array);
           })
      .def("decl_arr_arg",
           [&](Program *program, const DataType &dt, int total_dim,
               std::vector<int> shape) {
             return program->current_callable->insert_arr_arg(dt, total_dim,
                                                              shape);
           })
      .def("decl_ret", [&](Program *program, const DataType &dt) {
        return program->current_callable->insert_ret(dt);
      });

  py::class_<AotModuleBuilder>(m, "AotModuleBuilder")
      .def("add_field", &AotModuleBuilder::add_field)
      .def("add", &AotModuleBuilder::add)
      .def("add_kernel_template", &AotModuleBuilder::add_kernel_template)
      .def("dump", &AotModuleBuilder::dump);

  py::class_<Axis>(m, "Axis").def(py::init<int>());
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def_readwrite("parent", &SNode::parent)
      .def_readonly("type", &SNode::type)
      .def_readonly("id", &SNode::id)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &, bool))(&SNode::dense),
           py::return_value_policy::reference)
      .def(
          "pointer",
          (SNode & (SNode::*)(const std::vector<Axis> &,
                              const std::vector<int> &, bool))(&SNode::pointer),
          py::return_value_policy::reference)
      .def("hash",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &, bool))(&SNode::hash),
           py::return_value_policy::reference)
      .def("dynamic", &SNode::dynamic, py::return_value_policy::reference)
      .def("bitmasked",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               bool))(&SNode::bitmasked),
           py::return_value_policy::reference)
      .def("bit_struct", &SNode::bit_struct, py::return_value_policy::reference)
      .def("bit_array", &SNode::bit_array, py::return_value_policy::reference)
      .def("place", &SNode::place)
      .def("data_type", [](SNode *snode) { return snode->dt; })
      .def("name", [](SNode *snode) { return snode->name; })
      .def("get_num_ch",
           [](SNode *snode) -> int { return (int)snode->ch.size(); })
      .def(
          "get_ch",
          [](SNode *snode, int i) -> SNode * { return snode->ch[i].get(); },
          py::return_value_policy::reference)
      .def("lazy_grad", &SNode::lazy_grad)
      .def("read_int", &SNode::read_int)
      .def("read_uint", &SNode::read_uint)
      .def("read_float", &SNode::read_float)
      .def("has_grad", &SNode::has_grad)
      .def("is_primal", &SNode::is_primal)
      .def("is_place", &SNode::is_place)
      .def("get_expr", &SNode::get_expr)
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
           [](SNode *snode) { return snode->num_active_indices; })
      .def_readonly("cell_size_bytes", &SNode::cell_size_bytes)
      .def_readonly("offset_bytes_in_parent_cell",
                    &SNode::offset_bytes_in_parent_cell)
      .def("begin_shared_exp_placement", &SNode::begin_shared_exp_placement)
      .def("end_shared_exp_placement", &SNode::end_shared_exp_placement);

  py::class_<SNodeTree>(m, "SNodeTree")
      .def("id", &SNodeTree::id)
      .def("destroy_snode_tree", [](SNodeTree *snode_tree, Program *program) {
        program->destroy_snode_tree(snode_tree);
      });

  py::class_<Ndarray>(m, "Ndarray")
      .def(py::init<Program *, const DataType &, const std::vector<int> &>())
      .def("data_ptr", &Ndarray::get_data_ptr_as_int)
      .def("device_allocation_ptr", &Ndarray::get_device_allocation_ptr_as_int)
      .def("element_size", &Ndarray::get_element_size)
      .def("nelement", &Ndarray::get_nelement)
      .def("fill_float", &Ndarray::fill_float)
      .def("fill_int", &Ndarray::fill_int)
      .def("fill_uint", &Ndarray::fill_uint)
      .def("read_int", &Ndarray::read_int)
      .def("read_uint", &Ndarray::read_uint)
      .def("read_float", &Ndarray::read_float)
      .def("write_int", &Ndarray::write_int)
      .def("write_float", &Ndarray::write_float)
      .def_readonly("dtype", &Ndarray::dtype)
      .def_readonly("shape", &Ndarray::shape);

  py::class_<Kernel>(m, "Kernel")
      .def("get_ret_int", &Kernel::get_ret_int)
      .def("get_ret_float", &Kernel::get_ret_float)
      .def("make_launch_context", &Kernel::make_launch_context)
      .def(
          "ast_builder",
          [](Kernel *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference)
      .def("__call__",
           [](Kernel *kernel, Kernel::LaunchContextBuilder &launch_ctx) {
             py::gil_scoped_release release;
             kernel->operator()(launch_ctx);
           });

  py::class_<Kernel::LaunchContextBuilder>(m, "KernelLaunchContext")
      .def("set_arg_int", &Kernel::LaunchContextBuilder::set_arg_int)
      .def("set_arg_float", &Kernel::LaunchContextBuilder::set_arg_float)
      .def("set_arg_external_array",
           &Kernel::LaunchContextBuilder::set_arg_external_array)
      .def("set_arg_external_array_with_shape",
           &Kernel::LaunchContextBuilder::set_arg_external_array_with_shape)
      .def("set_arg_ndarray", &Kernel::LaunchContextBuilder::set_arg_ndarray)
      .def("set_extra_arg_int",
           &Kernel::LaunchContextBuilder::set_extra_arg_int);

  py::class_<Function>(m, "Function")
      .def("set_function_body",
           py::overload_cast<const std::function<void()> &>(
               &Function::set_function_body))
      .def(
          "ast_builder",
          [](Function *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<Expr> expr(m, "Expr");
  expr.def("serialize", [](Expr *expr) { return expr->serialize(); })
      .def("snode", &Expr::snode, py::return_value_policy::reference)
      .def("is_global_var",
           [](Expr *expr) { return expr->is<GlobalVariableExpression>(); })
      .def("is_external_var",
           [](Expr *expr) { return expr->is<ExternalTensorExpression>(); })
      .def("is_global_ptr",
           [](Expr *expr) { return expr->is<GlobalPtrExpression>(); })
      .def("is_primal",
           [](Expr *expr) {
             return expr->cast<GlobalVariableExpression>()->is_primal;
           })
      .def("set_tb", &Expr::set_tb)
      .def("set_name",
           [&](Expr *expr, std::string na) {
             expr->cast<GlobalVariableExpression>()->name = na;
           })
      .def("set_is_primal",
           [&](Expr *expr, bool v) {
             expr->cast<GlobalVariableExpression>()->is_primal = v;
           })
      .def("set_grad", &Expr::set_grad)
      .def("set_attribute", &Expr::set_attribute)
      .def("get_ret_type", &Expr::get_ret_type)
      .def("type_check", &Expr::type_check)
      .def("get_expr_name",
           [](Expr *expr) {
             return expr->cast<GlobalVariableExpression>()->name;
           })
      .def("get_attribute", &Expr::get_attribute)
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; })
      .def("get_underlying_ptr_address", [](Expr *e) {
        // The reason that there are both get_raw_address() and
        // get_underlying_ptr_address() is that Expr itself is mostly wrapper
        // around its underlying |expr| (of type Expression). Expr |e| can be
        // temporary, while the underlying |expr| is mostly persistant.
        //
        // Same get_raw_address() implies that get_underlying_ptr_address() are
        // also the same. The reverse is not true.
        return (uint64)e->expr.get();
      });

  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("size", [](ExprGroup *eg) { return eg->exprs.size(); })
      .def("push_back", &ExprGroup::push_back)
      .def("serialize", [](ExprGroup *eg) { eg->serialize(); });

  py::class_<Stmt>(m, "Stmt");

  m.def("expr_get_addr", [](SNode *snode, const ExprGroup &indices) {
    return Expr::make<SNodeOpExpression>(snode, SNodeOpType::get_addr, indices);
  });

  m.def("insert_append",
        [](SNode *snode, const ExprGroup &indices, const Expr &val) {
          return Append(snode, indices, val);
        });

  m.def("insert_is_active", [](SNode *snode, const ExprGroup &indices) {
    return is_active(snode, indices);
  });

  m.def("insert_len", [](SNode *snode, const ExprGroup &indices) {
    return Length(snode, indices);
  });

  m.def("insert_internal_func_call",
        [&](const std::string &func_name, const ExprGroup &args) {
          return Expr::make<InternalFuncCallExpression>(func_name, args.exprs);
        });

  m.def("make_func_call_expr",
        Expr::make<FuncCallExpression, Function *, const ExprGroup &>);

  m.def("value_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(cast));
  m.def("bits_cast",
        static_cast<Expr (*)(const Expr &expr, DataType)>(bit_cast));

  m.def("expr_atomic_add", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::add, a, b);
  });

  m.def("expr_atomic_sub", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::sub, a, b);
  });

  m.def("expr_atomic_min", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::min, a, b);
  });

  m.def("expr_atomic_max", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::max, a, b);
  });

  m.def("expr_atomic_bit_and", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_and, a, b);
  });

  m.def("expr_atomic_bit_or", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_or, a, b);
  });

  m.def("expr_atomic_bit_xor", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::bit_xor, a, b);
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
  m.def("expr_bit_shl", expr_bit_shl);
  m.def("expr_bit_shr", expr_bit_shr);
  m.def("expr_bit_sar", expr_bit_sar);
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

  m.def("expr_loop_unique", LoopUnique);

  m.def("expr_select", expr_select);

#define DEFINE_EXPRESSION_OP_UNARY(x) m.def("expr_" #x, expr_##x);

  m.def("expr_neg", [&](const Expr &e) { return -e; });
  DEFINE_EXPRESSION_OP_UNARY(sqrt)
  DEFINE_EXPRESSION_OP_UNARY(round)
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

  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>);
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>);
  m.def("make_frontend_assign_stmt",
        Stmt::make<FrontendAssignStmt, const Expr &, const Expr &>);

  m.def("make_arg_load_expr",
        Expr::make<ArgLoadExpression, int, const DataType &>);

  m.def("make_external_tensor_expr",
        Expr::make<ExternalTensorExpression, const DataType &, int, int, int>);

  m.def("make_id_expr", Expr::make<IdExpression, std::string>);

  m.def("make_rand_expr", Expr::make<RandExpression, const DataType &>);

  m.def("make_const_expr_int",
        Expr::make<ConstExpression, const DataType &, int64>);

  m.def("make_const_expr_fp",
        Expr::make<ConstExpression, const DataType &, float64>);

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
#define PER_TYPE(x)                                                  \
  m.attr(("DataType_" + data_type_name(PrimitiveType::x)).c_str()) = \
      PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

  m.def("data_type_size", data_type_size);
  m.def("is_custom_type", is_custom_type);
  m.def("is_integral", is_integral);
  m.def("is_signed", is_signed);
  m.def("is_real", is_real);
  m.def("is_unsigned", is_unsigned);

  m.def("global_new", static_cast<Expr (*)(Expr, DataType)>(global_new));
  m.def("set_global_grad", [&](const Expr &expr) {
    TI_ASSERT(expr.is<GlobalVariableExpression>());
    expr.cast<GlobalVariableExpression>()->is_primal = false;
  });
  m.def("data_type_name", data_type_name);

  m.def("subscript", [](const Expr &expr, const ExprGroup &expr_group) {
    return expr[expr_group];
  });

  m.def("make_tensor_element_expr",
        Expr::make<TensorElementExpression, const Expr &, const ExprGroup &,
                   const std::vector<int> &, int>);

  m.def("subscript", [](SNode *snode, const ExprGroup &indices) {
    return Expr::make<GlobalPtrExpression>(snode, indices);
  });

  m.def("get_external_tensor_dim", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->dim;
  });

  m.def("get_external_tensor_shape_along_axis",
        Expr::make<ExternalTensorShapeAlongAxisExpression, const Expr &, int>);

  // Mesh related.
  m.def("get_relation_size", [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx,
                                mesh::MeshElementType to_type) {
    return Expr::make<MeshRelationAccessExpression>(mesh_ptr.ptr.get(),
                                                    mesh_idx, to_type);
  });

  m.def("get_relation_access",
        [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx,
           mesh::MeshElementType to_type, const Expr &neighbor_idx) {
          return Expr::make<MeshRelationAccessExpression>(
              mesh_ptr.ptr.get(), mesh_idx, to_type, neighbor_idx);
        });

  m.def("get_index_conversion",
        [](mesh::MeshPtr mesh_ptr, mesh::MeshElementType idx_type,
           const Expr &idx, mesh::ConvType &conv_type) {
          return Expr::make<MeshIndexConversionExpression>(
              mesh_ptr.ptr.get(), idx_type, idx, conv_type);
        });

  py::class_<FunctionKey>(m, "FunctionKey")
      .def(py::init<const std::string &, int, int>())
      .def_readonly("instance_id", &FunctionKey::instance_id);

  m.def("test_throw", [] {
    try {
      throw IRModified();
    } catch (IRModified) {
      TI_INFO("caught");
    }
  });
  // Schedules
  m.def("parallelize", Parallelize);
  m.def("bit_vectorize", BitVectorize);
  m.def("block_dim", BlockDim);

  py::enum_<SNodeAccessFlag>(m, "SNodeAccessFlag", py::arithmetic())
      .value("block_local", SNodeAccessFlag::block_local)
      .value("read_only", SNodeAccessFlag::read_only)
      .value("mesh_local", SNodeAccessFlag::mesh_local)
      .export_values();

  m.def("insert_snode_access_flag", insert_snode_access_flag);
  m.def("reset_snode_access_flag", reset_snode_access_flag);

  m.def("test_throw", [] { throw IRModified(); });
  m.def("needs_grad", needs_grad);

#if TI_WITH_LLVM
  m.def("libdevice_path", libdevice_path);
#endif

  m.def("host_arch", host_arch);

  m.def("set_lib_dir", [&](const std::string &dir) { compiled_lib_dir = dir; });
  m.def("set_tmp_dir", [&](const std::string &dir) { runtime_tmp_dir = dir; });

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
#if TI_WITH_LLVM
  m.def("get_llvm_version_string", [] { return LLVM_VERSION_STRING; });
#endif
  m.def("test_printf", [] { printf("test_printf\n"); });
  m.def("test_logging", [] { TI_INFO("test_logging"); });
  m.def("trigger_crash", [] { *(int *)(1) = 0; });
  m.def("get_max_num_indices", [] { return taichi_max_num_indices; });
  m.def("get_max_num_args", [] { return taichi_max_num_args; });
  m.def("test_threading", test_threading);
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

  m.def("record_action_entry",
        [](std::string name,
           std::vector<std::pair<std::string,
                                 std::variant<std::string, int, float>>> args) {
          std::vector<ActionArg> acts;
          for (auto const &[k, v] : args) {
            if (std::holds_alternative<int>(v)) {
              acts.push_back(ActionArg(k, std::get<int>(v)));
            } else if (std::holds_alternative<float>(v)) {
              acts.push_back(ActionArg(k, std::get<float>(v)));
            } else {
              acts.push_back(ActionArg(k, std::get<std::string>(v)));
            }
          }
          ActionRecorder::get_instance().record(name, acts);
        });

  m.def("start_recording", [](const std::string &fn) {
    ActionRecorder::get_instance().start_recording(fn);
  });

  m.def("stop_recording",
        []() { ActionRecorder::get_instance().stop_recording(); });

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

  // Type system

  py::class_<Type>(m, "Type").def("to_string", &Type::to_string);

  // Note that it is important to specify py::return_value_policy::reference for
  // the factory methods, otherwise pybind11 will delete the Types owned by
  // TypeFactory on Python-scope pointer destruction.
  py::class_<TypeFactory>(m, "TypeFactory")
      .def("get_custom_int_type", &TypeFactory::get_custom_int_type,
           py::arg("num_bits"), py::arg("is_signed"), py::arg("compute_type"),
           py::return_value_policy::reference)
      .def("get_custom_float_type", &TypeFactory::get_custom_float_type,
           py::arg("digits_type"), py::arg("exponent_type"),
           py::arg("compute_type"), py::arg("scale"),
           py::return_value_policy::reference);

  m.def("get_type_factory_instance", TypeFactory::get_instance,
        py::return_value_policy::reference);

  py::class_<SNodeRegistry>(m, "SNodeRegistry")
      .def(py::init<>())
      .def("create_root", &SNodeRegistry::create_root,
           py::return_value_policy::reference);
  m.def(
      "finalize_snode_tree",
      [](SNodeRegistry *registry, const SNode *root, Program *program,
         bool compile_only) -> SNodeTree * {
        return program->add_snode_tree(registry->finalize(root), compile_only);
      },
      py::return_value_policy::reference);

  py::class_<SparseMatrixBuilder>(m, "SparseMatrixBuilder")
      .def("print_triplets", &SparseMatrixBuilder::print_triplets)
      .def("build", &SparseMatrixBuilder::build)
      .def("get_addr", [](SparseMatrixBuilder *mat) { return uint64(mat); });

  py::class_<SparseMatrix>(m, "SparseMatrix")
      .def("to_string", &SparseMatrix::to_string)
      .def(py::self + py::self, py::return_value_policy::reference_internal)
      .def(py::self - py::self, py::return_value_policy::reference_internal)
      .def(float() * py::self, py::return_value_policy::reference_internal)
      .def(py::self * float(), py::return_value_policy::reference_internal)
      .def(py::self * py::self, py::return_value_policy::reference_internal)
      .def("matmul", &SparseMatrix::matmul,
           py::return_value_policy::reference_internal)
      .def("mat_vec_mul", &SparseMatrix::mat_vec_mul)
      .def("transpose", &SparseMatrix::transpose,
           py::return_value_policy::reference_internal)
      .def("get_element", &SparseMatrix::get_element)
      .def("set_element", &SparseMatrix::set_element)
      .def("num_rows", &SparseMatrix::num_rows)
      .def("num_cols", &SparseMatrix::num_cols);

  py::class_<SparseSolver>(m, "SparseSolver")
      .def("compute", &SparseSolver::compute)
      .def("analyze_pattern", &SparseSolver::analyze_pattern)
      .def("factorize", &SparseSolver::factorize)
      .def("solve", &SparseSolver::solve)
      .def("info", &SparseSolver::info);

  m.def("make_sparse_solver", &make_sparse_solver);

  // Mesh Class
  // Mesh related.
  py::enum_<mesh::MeshTopology>(m, "MeshTopology", py::arithmetic())
      .value("Triangle", mesh::MeshTopology::Triangle)
      .value("Tetrahedron", mesh::MeshTopology::Tetrahedron)
      .export_values();

  py::enum_<mesh::MeshElementType>(m, "MeshElementType", py::arithmetic())
      .value("Vertex", mesh::MeshElementType::Vertex)
      .value("Edge", mesh::MeshElementType::Edge)
      .value("Face", mesh::MeshElementType::Face)
      .value("Cell", mesh::MeshElementType::Cell)
      .export_values();

  py::enum_<mesh::MeshRelationType>(m, "MeshRelationType", py::arithmetic())
      .value("VV", mesh::MeshRelationType::VV)
      .value("VE", mesh::MeshRelationType::VE)
      .value("VF", mesh::MeshRelationType::VF)
      .value("VC", mesh::MeshRelationType::VC)
      .value("EV", mesh::MeshRelationType::EV)
      .value("EE", mesh::MeshRelationType::EE)
      .value("EF", mesh::MeshRelationType::EF)
      .value("EC", mesh::MeshRelationType::EC)
      .value("FV", mesh::MeshRelationType::FV)
      .value("FE", mesh::MeshRelationType::FE)
      .value("FF", mesh::MeshRelationType::FF)
      .value("FC", mesh::MeshRelationType::FC)
      .value("CV", mesh::MeshRelationType::CV)
      .value("CE", mesh::MeshRelationType::CE)
      .value("CF", mesh::MeshRelationType::CF)
      .value("CC", mesh::MeshRelationType::CC)
      .export_values();

  py::enum_<mesh::ConvType>(m, "ConvType", py::arithmetic())
      .value("l2g", mesh::ConvType::l2g)
      .value("l2r", mesh::ConvType::l2r)
      .value("g2r", mesh::ConvType::g2r)
      .export_values();

  py::class_<mesh::Mesh>(m, "Mesh");
  py::class_<mesh::MeshPtr>(m, "MeshPtr");

  m.def("element_order", mesh::element_order);
  m.def("from_end_element_order", mesh::from_end_element_order);
  m.def("to_end_element_order", mesh::to_end_element_order);
  m.def("relation_by_orders", mesh::relation_by_orders);
  m.def("inverse_relation", mesh::inverse_relation);
  m.def("element_type_name", mesh::element_type_name);

  m.def(
      "create_mesh",
      []() {
        auto mesh_shared = std::make_shared<mesh::Mesh>();
        mesh::MeshPtr mesh_ptr = mesh::MeshPtr{mesh_shared};
        return mesh_ptr;
      },
      py::return_value_policy::reference);

  // ad-hoc setters
  m.def("set_owned_offset",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
          mesh_ptr.ptr->owned_offset.insert(std::pair(type, snode));
        });
  m.def("set_total_offset",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
          mesh_ptr.ptr->total_offset.insert(std::pair(type, snode));
        });
  m.def("set_num_patches", [](mesh::MeshPtr &mesh_ptr, int num_patches) {
    mesh_ptr.ptr->num_patches = num_patches;
  });

  m.def("set_num_elements", [](mesh::MeshPtr &mesh_ptr,
                               mesh::MeshElementType type, int num_elements) {
    mesh_ptr.ptr->num_elements.insert(std::pair(type, num_elements));
  });

  m.def("get_num_elements",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type) {
          return mesh_ptr.ptr->num_elements.find(type)->second;
        });

  m.def("set_patch_max_element_num",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type,
           int max_element_num) {
          mesh_ptr.ptr->patch_max_element_num.insert(
              std::pair(type, max_element_num));
        });

  m.def("set_index_mapping",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType element_type,
           mesh::ConvType conv_type, SNode *snode) {
          mesh_ptr.ptr->index_mapping.insert(
              std::make_pair(std::make_pair(element_type, conv_type), snode));
        });

  m.def("set_relation_fixed",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value) {
          mesh_ptr.ptr->relations.insert(
              std::pair(type, mesh::MeshLocalRelation(value)));
        });

  m.def("set_relation_dynamic",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value,
           SNode *offset) {
          mesh_ptr.ptr->relations.insert(
              std::pair(type, mesh::MeshLocalRelation(value, offset)));
        });
}

TI_NAMESPACE_END
