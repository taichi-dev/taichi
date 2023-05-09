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

#include "taichi/ir/expression_ops.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/graph_builder.h"
#include "taichi/program/extension.h"
#include "taichi/program/ndarray.h"
#include "taichi/python/export.h"
#include "taichi/math/svd.h"
#include "taichi/system/timeline.h"
#include "taichi/python/snode_registry.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/sparse_solver.h"
#include "taichi/program/conjugate_gradient.h"
#include "taichi/aot/graph_data.h"
#include "taichi/ir/mesh.h"

#include "taichi/program/kernel_profiler.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

namespace taichi {
bool test_threading();

}  // namespace taichi

namespace taichi::lang {

std::string libdevice_path();

}  // namespace taichi::lang

namespace taichi {
void export_lang(py::module &m) {
  using namespace taichi::lang;
  using namespace std::placeholders;

  py::register_exception<TaichiTypeError>(m, "TaichiTypeError",
                                          PyExc_TypeError);
  py::register_exception<TaichiSyntaxError>(m, "TaichiSyntaxError",
                                            PyExc_SyntaxError);
  py::register_exception<TaichiIndexError>(m, "TaichiIndexError",
                                           PyExc_IndexError);
  py::register_exception<TaichiRuntimeError>(m, "TaichiRuntimeError",
                                             PyExc_RuntimeError);
  py::register_exception<TaichiAssertionError>(m, "TaichiAssertionError",
                                               PyExc_AssertionError);
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

  py::enum_<ExternalArrayLayout>(m, "Layout", py::arithmetic())
      .value("AOS", ExternalArrayLayout::kAOS)
      .value("SOA", ExternalArrayLayout::kSOA)
      .value("NULL", ExternalArrayLayout::kNull)
      .export_values();

  py::enum_<AutodiffMode>(m, "AutodiffMode", py::arithmetic())
      .value("NONE", AutodiffMode::kNone)
      .value("VALIDATION", AutodiffMode::kCheckAutodiffValid)
      .value("FORWARD", AutodiffMode::kForward)
      .value("REVERSE", AutodiffMode::kReverse)
      .export_values();

  py::enum_<SNodeGradType>(m, "SNodeGradType", py::arithmetic())
      .value("PRIMAL", SNodeGradType::kPrimal)
      .value("ADJOINT", SNodeGradType::kAdjoint)
      .value("DUAL", SNodeGradType::kDual)
      .value("ADJOINT_CHECKBIT", SNodeGradType::kAdjointCheckbit)
      .export_values();

  // TODO(type): This should be removed
  py::class_<DataType>(m, "DataType")
      .def(py::init<Type *>())
      .def(py::self == py::self)
      .def("__hash__", &DataType::hash)
      .def("to_string", &DataType::to_string)
      .def("__str__", &DataType::to_string)
      .def("shape", &DataType::get_shape)
      .def("element_type", &DataType::get_element_type)
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
      .def_readwrite("print_ir", &CompileConfig::print_ir)
      .def_readwrite("print_preprocessed_ir",
                     &CompileConfig::print_preprocessed_ir)
      .def_readwrite("debug", &CompileConfig::debug)
      .def_readwrite("cfg_optimization", &CompileConfig::cfg_optimization)
      .def_readwrite("check_out_of_bound", &CompileConfig::check_out_of_bound)
      .def_readwrite("print_accessor_ir", &CompileConfig::print_accessor_ir)
      .def_readwrite("use_llvm", &CompileConfig::use_llvm)
      .def_readwrite("print_struct_llvm_ir",
                     &CompileConfig::print_struct_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir",
                     &CompileConfig::print_kernel_llvm_ir)
      .def_readwrite("print_kernel_llvm_ir_optimized",
                     &CompileConfig::print_kernel_llvm_ir_optimized)
      .def_readwrite("print_kernel_asm", &CompileConfig::print_kernel_asm)
      .def_readwrite("print_kernel_amdgcn", &CompileConfig::print_kernel_amdgcn)
      .def_readwrite("simplify_before_lower_access",
                     &CompileConfig::simplify_before_lower_access)
      .def_readwrite("simplify_after_lower_access",
                     &CompileConfig::simplify_after_lower_access)
      .def_readwrite("lower_access", &CompileConfig::lower_access)
      .def_readwrite("move_loop_invariant_outside_if",
                     &CompileConfig::move_loop_invariant_outside_if)
      .def_readwrite("cache_loop_invariant_global_vars",
                     &CompileConfig::cache_loop_invariant_global_vars)
      .def_readwrite("default_cpu_block_dim",
                     &CompileConfig::default_cpu_block_dim)
      .def_readwrite("cpu_block_dim_adaptive",
                     &CompileConfig::cpu_block_dim_adaptive)
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
      .def_readwrite("default_up", &CompileConfig::default_up)
      .def_readwrite("device_memory_GB", &CompileConfig::device_memory_GB)
      .def_readwrite("device_memory_fraction",
                     &CompileConfig::device_memory_fraction)
      .def_readwrite("fast_math", &CompileConfig::fast_math)
      .def_readwrite("advanced_optimization",
                     &CompileConfig::advanced_optimization)
      .def_readwrite("ad_stack_size", &CompileConfig::ad_stack_size)
      .def_readwrite("flatten_if", &CompileConfig::flatten_if)
      .def_readwrite("make_thread_local", &CompileConfig::make_thread_local)
      .def_readwrite("make_block_local", &CompileConfig::make_block_local)
      .def_readwrite("detect_read_only", &CompileConfig::detect_read_only)
      .def_readwrite("real_matrix_scalarize",
                     &CompileConfig::real_matrix_scalarize)
      .def_readwrite("half2_vectorization", &CompileConfig::half2_vectorization)
      .def_readwrite("make_cpu_multithreading_loop",
                     &CompileConfig::make_cpu_multithreading_loop)
      .def_readwrite("quant_opt_store_fusion",
                     &CompileConfig::quant_opt_store_fusion)
      .def_readwrite("quant_opt_atomic_demotion",
                     &CompileConfig::quant_opt_atomic_demotion)
      .def_readwrite("allow_nv_shader_extension",
                     &CompileConfig::allow_nv_shader_extension)
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
                     &CompileConfig::auto_mesh_local_default_occupacy)
      .def_readwrite("offline_cache", &CompileConfig::offline_cache)
      .def_readwrite("offline_cache_file_path",
                     &CompileConfig::offline_cache_file_path)
      .def_readwrite("offline_cache_cleaning_policy",
                     &CompileConfig::offline_cache_cleaning_policy)
      .def_readwrite("offline_cache_max_size_of_files",
                     &CompileConfig::offline_cache_max_size_of_files)
      .def_readwrite("offline_cache_cleaning_factor",
                     &CompileConfig::offline_cache_cleaning_factor)
      .def_readwrite("num_compile_threads", &CompileConfig::num_compile_threads)
      .def_readwrite("vk_api_version", &CompileConfig::vk_api_version)
      .def_readwrite("cuda_stack_limit", &CompileConfig::cuda_stack_limit);

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

  py::enum_<SNodeAccessFlag>(m, "SNodeAccessFlag", py::arithmetic())
      .value("block_local", SNodeAccessFlag::block_local)
      .value("read_only", SNodeAccessFlag::read_only)
      .value("mesh_local", SNodeAccessFlag::mesh_local)
      .export_values();

  // Export ASTBuilder
  py::class_<ASTBuilder>(m, "ASTBuilder")
      .def("make_id_expr", &ASTBuilder::make_id_expr)
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
      .def("insert_deactivate", &ASTBuilder::insert_snode_deactivate)
      .def("insert_activate", &ASTBuilder::insert_snode_activate)
      .def("expr_snode_get_addr", &ASTBuilder::snode_get_addr)
      .def("expr_snode_append", &ASTBuilder::snode_append)
      .def("expr_snode_is_active", &ASTBuilder::snode_is_active)
      .def("expr_snode_length", &ASTBuilder::snode_length)
      .def("insert_external_func_call", &ASTBuilder::insert_external_func_call)
      .def("make_matrix_expr", &ASTBuilder::make_matrix_expr)
      .def("expr_alloca", &ASTBuilder::expr_alloca)
      .def("expr_alloca_shared_array", &ASTBuilder::expr_alloca_shared_array)
      .def("create_assert_stmt", &ASTBuilder::create_assert_stmt)
      .def("expr_assign", &ASTBuilder::expr_assign)
      .def("begin_frontend_range_for", &ASTBuilder::begin_frontend_range_for)
      .def("end_frontend_range_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_struct_for_on_snode",
           &ASTBuilder::begin_frontend_struct_for_on_snode)
      .def("begin_frontend_struct_for_on_external_tensor",
           &ASTBuilder::begin_frontend_struct_for_on_external_tensor)
      .def("end_frontend_struct_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_mesh_for", &ASTBuilder::begin_frontend_mesh_for)
      .def("end_frontend_mesh_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_while", &ASTBuilder::begin_frontend_while)
      .def("insert_break_stmt", &ASTBuilder::insert_break_stmt)
      .def("insert_continue_stmt", &ASTBuilder::insert_continue_stmt)
      .def("insert_expr_stmt", &ASTBuilder::insert_expr_stmt)
      .def("insert_thread_idx_expr", &ASTBuilder::insert_thread_idx_expr)
      .def("insert_patch_idx_expr", &ASTBuilder::insert_patch_idx_expr)
      .def("make_texture_op_expr", &ASTBuilder::make_texture_op_expr)
      .def("expand_exprs", &ASTBuilder::expand_exprs)
      .def("mesh_index_conversion", &ASTBuilder::mesh_index_conversion)
      .def("expr_subscript", &ASTBuilder::expr_subscript)
      .def("insert_func_call", &ASTBuilder::insert_func_call)
      .def("sifakis_svd_f32", sifakis_svd_export<float32, int32>)
      .def("sifakis_svd_f64", sifakis_svd_export<float64, int64>)
      .def("expr_var", &ASTBuilder::make_var)
      .def("bit_vectorize", &ASTBuilder::bit_vectorize)
      .def("parallelize", &ASTBuilder::parallelize)
      .def("strictly_serialize", &ASTBuilder::strictly_serialize)
      .def("block_dim", &ASTBuilder::block_dim)
      .def("insert_snode_access_flag", &ASTBuilder::insert_snode_access_flag)
      .def("reset_snode_access_flag", &ASTBuilder::reset_snode_access_flag);

  py::class_<DeviceCapabilityConfig>(
      m, "DeviceCapabilityConfig");  // NOLINT(bugprone-unused-raii)

  py::class_<CompiledKernelData>(
      m, "CompiledKernelData");  // NOLINT(bugprone-unused-raii)

  py::class_<Program>(m, "Program")
      .def(py::init<>())
      .def("config", &Program::compile_config,
           py::return_value_policy::reference)
      .def("sync_kernel_profiler",
           [](Program *program) { program->profiler->sync(); })
      .def("update_kernel_profiler",
           [](Program *program) { program->profiler->update(); })
      .def("clear_kernel_profiler",
           [](Program *program) { program->profiler->clear(); })
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
      .def("timeline_clear",
           [](Program *) { Timelines::get_instance().clear(); })
      .def("timeline_save",
           [](Program *, const std::string &fn) {
             Timelines::get_instance().save(fn);
           })
      .def("print_memory_profiler_info", &Program::print_memory_profiler_info)
      .def("finalize", &Program::finalize)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("get_snode_num_dynamically_allocated",
           &Program::get_snode_num_dynamically_allocated)
      .def("synchronize", &Program::synchronize)
      .def("materialize_runtime", &Program::materialize_runtime)
      .def("make_aot_module_builder", &Program::make_aot_module_builder)
      .def("get_snode_tree_size", &Program::get_snode_tree_size)
      .def("get_snode_root", &Program::get_snode_root,
           py::return_value_policy::reference)
      .def(
          "create_kernel",
          [](Program *program, const std::function<void(Kernel *)> &body,
             const std::string &name, AutodiffMode autodiff_mode) -> Kernel * {
            py::gil_scoped_release release;
            return &program->kernel(body, name, autodiff_mode);
          },
          py::return_value_policy::reference)
      .def("create_function", &Program::create_function,
           py::return_value_policy::reference)
      .def("create_sparse_matrix",
           [](Program *program, int n, int m, DataType dtype,
              std::string storage_format) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                             !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             if (arch_is_cpu(program->compile_config().arch))
               return make_sparse_matrix(n, m, dtype, storage_format);
             else
               return make_cu_sparse_matrix(n, m, dtype);
           })
      .def("make_sparse_matrix_from_ndarray",
           [](Program *program, SparseMatrix &sm, const Ndarray &ndarray) {
             TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                             !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             return make_sparse_matrix_from_ndarray(program, sm, ndarray);
           })
      .def("make_id_expr",
           [](Program *program, const std::string &name) {
             return Expr::make<IdExpression>(program->get_next_global_id(name));
           })
      .def(
          "create_ndarray",
          [&](Program *program, const DataType &dt,
              const std::vector<int> &shape, ExternalArrayLayout layout,
              bool zero_fill) -> Ndarray * {
            return program->create_ndarray(dt, shape, layout, zero_fill);
          },
          py::arg("dt"), py::arg("shape"),
          py::arg("layout") = ExternalArrayLayout::kNull,
          py::arg("zero_fill") = false, py::return_value_policy::reference)
      .def("delete_ndarray", &Program::delete_ndarray)
      .def(
          "create_texture",
          [&](Program *program, BufferFormat fmt, const std::vector<int> &shape)
              -> Texture * { return program->create_texture(fmt, shape); },
          py::arg("fmt"), py::arg("shape") = py::tuple(),
          py::return_value_policy::reference)
      .def("get_ndarray_data_ptr_as_int",
           [](Program *program, Ndarray *ndarray) {
             return program->get_ndarray_data_ptr_as_int(ndarray);
           })
      .def("fill_float",
           [](Program *program, Ndarray *ndarray, float val) {
             program->fill_ndarray_fast_u32(ndarray,
                                            reinterpret_cast<uint32_t &>(val));
           })
      .def("fill_int",
           [](Program *program, Ndarray *ndarray, int32_t val) {
             program->fill_ndarray_fast_u32(ndarray,
                                            reinterpret_cast<int32_t &>(val));
           })
      .def("fill_uint",
           [](Program *program, Ndarray *ndarray, uint32_t val) {
             program->fill_ndarray_fast_u32(ndarray, val);
           })
      .def("get_graphics_device",
           [](Program *program) { return program->get_graphics_device(); })
      .def("compile_kernel", &Program::compile_kernel,
           py::return_value_policy::reference)
      .def("launch_kernel", &Program::launch_kernel)
      .def("get_device_caps", &Program::get_device_caps);

  py::class_<AotModuleBuilder>(m, "AotModuleBuilder")
      .def("add_field", &AotModuleBuilder::add_field)
      .def("add", &AotModuleBuilder::add)
      .def("add_kernel_template", &AotModuleBuilder::add_kernel_template)
      .def("add_graph", &AotModuleBuilder::add_graph)
      .def("dump", &AotModuleBuilder::dump);

  py::class_<Axis>(m, "Axis").def(py::init<int>());
  py::class_<SNode>(m, "SNode")
      .def(py::init<>())
      .def_readwrite("parent", &SNode::parent)
      .def_readonly("type", &SNode::type)
      .def_readonly("id", &SNode::id)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const std::string &))(&SNode::dense),
           py::return_value_policy::reference)
      .def("pointer",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const std::string &))(&SNode::pointer),
           py::return_value_policy::reference)
      .def("hash",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const std::string &))(&SNode::hash),
           py::return_value_policy::reference)
      .def("dynamic", &SNode::dynamic, py::return_value_policy::reference)
      .def("bitmasked",
           (SNode & (SNode::*)(const std::vector<Axis> &,
                               const std::vector<int> &,
                               const std::string &))(&SNode::bitmasked),
           py::return_value_policy::reference)
      .def("bit_struct", &SNode::bit_struct, py::return_value_policy::reference)
      .def("quant_array", &SNode::quant_array,
           py::return_value_policy::reference)
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
      .def("lazy_dual", &SNode::lazy_dual)
      .def("allocate_adjoint_checkbit", &SNode::allocate_adjoint_checkbit)
      .def("read_int", &SNode::read_int)
      .def("read_uint", &SNode::read_uint)
      .def("read_float", &SNode::read_float)
      .def("has_adjoint", &SNode::has_adjoint)
      .def("has_adjoint_checkbit", &SNode::has_adjoint_checkbit)
      .def("get_snode_grad_type", &SNode::get_snode_grad_type)
      .def("has_dual", &SNode::has_dual)
      .def("is_primal", &SNode::is_primal)
      .def("is_place", &SNode::is_place)
      .def("get_expr", &SNode::get_expr)
      .def("write_int", &SNode::write_int)
      .def("write_uint", &SNode::write_uint)
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
                    &SNode::offset_bytes_in_parent_cell);

  py::class_<SNodeTree>(m, "SNodeTree")
      .def("id", &SNodeTree::id)
      .def("destroy_snode_tree", [](SNodeTree *snode_tree, Program *program) {
        program->destroy_snode_tree(snode_tree);
      });

  py::class_<DeviceAllocation>(m, "DeviceAllocation")
      .def(py::init([](uint64_t device, uint64_t alloc_id) -> DeviceAllocation {
             DeviceAllocation alloc;
             alloc.device = (Device *)device;
             alloc.alloc_id = (DeviceAllocationId)alloc_id;
             return alloc;
           }),
           py::arg("device"), py::arg("alloc_id"))
      .def_readonly("device", &DeviceAllocation::device)
      .def_readonly("alloc_id", &DeviceAllocation::alloc_id);

  py::class_<Ndarray>(m, "Ndarray")
      .def("device_allocation_ptr", &Ndarray::get_device_allocation_ptr_as_int)
      .def("device_allocation", &Ndarray::get_device_allocation)
      .def("element_size", &Ndarray::get_element_size)
      .def("nelement", &Ndarray::get_nelement)
      .def("read_int", &Ndarray::read_int)
      .def("read_uint", &Ndarray::read_uint)
      .def("read_float", &Ndarray::read_float)
      .def("write_int", &Ndarray::write_int)
      .def("write_float", &Ndarray::write_float)
      .def("total_shape", &Ndarray::total_shape)
      .def("element_shape", &Ndarray::get_element_shape)
      .def("element_data_type", &Ndarray::get_element_data_type)
      .def_readonly("dtype", &Ndarray::dtype)
      .def_readonly("shape", &Ndarray::shape);

  py::enum_<BufferFormat>(m, "Format")
#define PER_BUFFER_FORMAT(x) .value(#x, BufferFormat::x)
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_EXTENSION
      ;

  py::class_<Texture>(m, "Texture")
      .def("device_allocation_ptr", &Texture::get_device_allocation_ptr_as_int)
      .def("from_ndarray", &Texture::from_ndarray)
      .def("from_snode", &Texture::from_snode);

  py::enum_<aot::ArgKind>(m, "ArgKind")
      .value("SCALAR", aot::ArgKind::kScalar)
      .value("NDARRAY", aot::ArgKind::kNdarray)
      // Using this MATRIX as Scalar alias, we can move to native matrix type
      // when supported
      .value("MATRIX", aot::ArgKind::kMatrix)
      .value("TEXTURE", aot::ArgKind::kTexture)
      .value("RWTEXTURE", aot::ArgKind::kRWTexture)
      .export_values();

  py::class_<aot::Arg>(m, "Arg")
      .def(py::init<aot::ArgKind, std::string, DataType &, size_t,
                    std::vector<int>>(),
           py::arg("tag"), py::arg("name"), py::arg("dtype"),
           py::arg("field_dim"), py::arg("element_shape"))
      .def(py::init<aot::ArgKind, std::string, DataType &, size_t,
                    std::vector<int>>(),
           py::arg("tag"), py::arg("name"), py::arg("channel_format"),
           py::arg("num_channels"), py::arg("shape"))
      .def_readonly("name", &aot::Arg::name)
      .def_readonly("element_shape", &aot::Arg::element_shape)
      .def_readonly("texture_shape", &aot::Arg::element_shape)
      .def_readonly("field_dim", &aot::Arg::field_dim)
      .def_readonly("num_channels", &aot::Arg::num_channels)
      .def("dtype", &aot::Arg::dtype)
      .def("channel_format", &aot::Arg::dtype);

  py::class_<Node>(m, "Node");  // NOLINT(bugprone-unused-raii)

  py::class_<Sequential, Node>(m, "Sequential")
      .def(py::init<GraphBuilder *>())
      .def("append", &Sequential::append)
      .def("dispatch", &Sequential::dispatch);

  py::class_<GraphBuilder>(m, "GraphBuilder")
      .def(py::init<>())
      .def("dispatch", &GraphBuilder::dispatch)
      .def("compile", &GraphBuilder::compile)
      .def("create_sequential", &GraphBuilder::new_sequential_node,
           py::return_value_policy::reference)
      .def("seq", &GraphBuilder::seq, py::return_value_policy::reference);

  py::class_<aot::CompiledGraph>(m, "CompiledGraph")
      .def("jit_run",
           [](aot::CompiledGraph *self, const CompileConfig &compile_config,
              const py::dict &pyargs) {
             std::unordered_map<std::string, aot::IValue> args;
             auto insert_scalar_arg = [&args](std::string arg_name,
                                              DataType expected_dtype,
                                              py::object pyarg) {
               auto type_id = expected_dtype->as<PrimitiveType>()->type;
               switch (type_id) {
#define PER_C_TYPE(type, ctype)                                           \
  case PrimitiveTypeID::type:                                             \
    args.insert({arg_name, aot::IValue::create(py::cast<ctype>(pyarg))}); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
                 default:
                   TI_ERROR("Unsupported scalar type {}", type_id);
               }
             };
             for (const auto &[arg_name, arg] : self->args) {
               auto tag = arg.tag;
               if (tag == aot::ArgKind::kMatrix) {
                 int size = arg.element_shape[0] * arg.element_shape[1];
                 for (int i = 0; i < size; i++) {
                   auto name = fmt::format("{}_{}", arg_name, i);
                   TI_ASSERT(pyargs.contains(name.c_str()));
                   auto pyarg = pyargs[name.c_str()];
                   insert_scalar_arg(name, arg.dtype(), pyarg);
                 }
                 continue;
               }
               TI_ASSERT(pyargs.contains(arg_name.c_str()));
               auto pyarg = pyargs[arg_name.c_str()];
               if (tag == aot::ArgKind::kNdarray) {
                 auto &val = pyarg.cast<Ndarray &>();
                 args.insert({arg_name, aot::IValue::create(val)});
               } else if (tag == aot::ArgKind::kTexture ||
                          tag == aot::ArgKind::kRWTexture) {
                 auto &val = pyarg.cast<Texture &>();
                 args.insert({arg_name, aot::IValue::create(val)});
               } else if (tag == aot::ArgKind::kScalar) {
                 auto expected_dtype = arg.dtype();
                 insert_scalar_arg(arg_name, expected_dtype, pyarg);
               } else {
                 TI_NOT_IMPLEMENTED;
               }
             }
             self->jit_run(compile_config, args);
           });

  py::class_<Kernel>(m, "Kernel")
      .def("no_activate",
           [](Kernel *self, SNode *snode) {
             // TODO(#2193): Also apply to @ti.func?
             self->no_activate.push_back(snode);
           })
      .def("insert_scalar_param", &Kernel::insert_scalar_param)
      .def("insert_arr_param", &Kernel::insert_arr_param)
      .def("insert_ndarray_param", &Kernel::insert_ndarray_param)
      .def("insert_texture_param", &Kernel::insert_texture_param)
      .def("insert_pointer_param", &Kernel::insert_pointer_param)
      .def("insert_rw_texture_param", &Kernel::insert_rw_texture_param)
      .def("insert_ret", &Kernel::insert_ret)
      .def("finalize_rets", &Kernel::finalize_rets)
      .def("finalize_params", &Kernel::finalize_params)
      .def("get_ret_int", &Kernel::get_ret_int)
      .def("get_ret_uint", &Kernel::get_ret_uint)
      .def("get_ret_float", &Kernel::get_ret_float)
      .def("get_ret_int_tensor", &Kernel::get_ret_int_tensor)
      .def("get_ret_uint_tensor", &Kernel::get_ret_uint_tensor)
      .def("get_ret_float_tensor", &Kernel::get_ret_float_tensor)
      .def("make_launch_context", &Kernel::make_launch_context)
      .def(
          "ast_builder",
          [](Kernel *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<LaunchContextBuilder>(m, "KernelLaunchContext")
      .def("set_arg_int", &LaunchContextBuilder::set_arg_int)
      .def("set_arg_uint", &LaunchContextBuilder::set_arg_uint)
      .def("set_arg_float", &LaunchContextBuilder::set_arg_float)
      .def("set_struct_arg_int", &LaunchContextBuilder::set_struct_arg<int64>)
      .def("set_struct_arg_uint", &LaunchContextBuilder::set_struct_arg<uint64>)
      .def("set_struct_arg_float",
           &LaunchContextBuilder::set_struct_arg<double>)
      .def("set_arg_external_array_with_shape",
           &LaunchContextBuilder::set_arg_external_array_with_shape)
      .def("set_arg_ndarray", &LaunchContextBuilder::set_arg_ndarray)
      .def("set_arg_ndarray_with_grad",
           &LaunchContextBuilder::set_arg_ndarray_with_grad)
      .def("set_arg_texture", &LaunchContextBuilder::set_arg_texture)
      .def("set_arg_rw_texture", &LaunchContextBuilder::set_arg_rw_texture)
      .def("get_struct_ret_int", &LaunchContextBuilder::get_struct_ret_int)
      .def("get_struct_ret_uint", &LaunchContextBuilder::get_struct_ret_uint)
      .def("get_struct_ret_float", &LaunchContextBuilder::get_struct_ret_float);

  py::class_<Function>(m, "Function")
      .def("insert_scalar_param", &Function::insert_scalar_param)
      .def("insert_arr_param", &Function::insert_arr_param)
      .def("insert_texture_param", &Function::insert_texture_param)
      .def("insert_pointer_param", &Function::insert_pointer_param)
      .def("insert_rw_texture_param", &Function::insert_rw_texture_param)
      .def("insert_ret", &Function::insert_ret)
      .def("set_function_body",
           py::overload_cast<const std::function<void()> &>(
               &Function::set_function_body))
      .def("finalize_rets", &Function::finalize_rets)
      .def("finalize_params", &Function::finalize_params)
      .def(
          "ast_builder",
          [](Function *self) -> ASTBuilder * {
            return &self->context->builder();
          },
          py::return_value_policy::reference);

  py::class_<Expr> expr(m, "Expr");
  expr.def("snode", &Expr::snode, py::return_value_policy::reference)
      .def("is_external_tensor_expr",
           [](Expr *expr) { return expr->is<ExternalTensorExpression>(); })
      .def("is_index_expr",
           [](Expr *expr) { return expr->is<IndexExpression>(); })
      .def("is_primal",
           [](Expr *expr) {
             return expr->cast<FieldExpression>()->snode_grad_type ==
                    SNodeGradType::kPrimal;
           })
      .def("is_lvalue", [](Expr *expr) { return expr->expr->is_lvalue(); })
      .def("set_tb", &Expr::set_tb)
      .def("set_name",
           [&](Expr *expr, std::string na) {
             expr->cast<FieldExpression>()->name = na;
           })
      .def("set_grad_type",
           [&](Expr *expr, SNodeGradType t) {
             expr->cast<FieldExpression>()->snode_grad_type = t;
           })
      .def("set_adjoint", &Expr::set_adjoint)
      .def("set_adjoint_checkbit", &Expr::set_adjoint_checkbit)
      .def("set_dual", &Expr::set_dual)
      .def("set_dynamic_index_stride",
           [&](Expr *expr, int dynamic_index_stride) {
             auto matrix_field = expr->cast<MatrixFieldExpression>();
             matrix_field->dynamic_indexable = true;
             matrix_field->dynamic_index_stride = dynamic_index_stride;
           })
      .def("get_dynamic_indexable",
           [&](Expr *expr) -> bool {
             return expr->cast<MatrixFieldExpression>()->dynamic_indexable;
           })
      .def("get_dynamic_index_stride",
           [&](Expr *expr) -> int {
             return expr->cast<MatrixFieldExpression>()->dynamic_index_stride;
           })
      .def(
          "get_dt",
          [&](Expr *expr) -> const Type * {
            return expr->cast<FieldExpression>()->dt;
          },
          py::return_value_policy::reference)
      .def("get_ret_type", &Expr::get_ret_type)
      .def("is_tensor",
           [](Expr *expr) { return expr->expr->ret_type->is<TensorType>(); })
      .def("is_struct",
           [](Expr *expr) {
             return expr->expr->ret_type.ptr_removed()->is<StructType>();
           })
      .def("get_shape",
           [](Expr *expr) -> std::optional<std::vector<int>> {
             if (expr->expr->ret_type->is<TensorType>()) {
               return std::optional<std::vector<int>>(
                   expr->expr->ret_type->cast<TensorType>()->get_shape());
             }
             return std::nullopt;
           })
      .def("type_check", &Expr::type_check)
      .def("get_expr_name",
           [](Expr *expr) { return expr->cast<FieldExpression>()->name; })
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; })
      .def("get_underlying_ptr_address", [](Expr *e) {
        // The reason that there are both get_raw_address() and
        // get_underlying_ptr_address() is that Expr itself is mostly wrapper
        // around its underlying |expr| (of type Expression). Expr |e| can be
        // temporary, while the underlying |expr| is mostly persistent.
        //
        // Same get_raw_address() implies that get_underlying_ptr_address() are
        // also the same. The reverse is not true.
        return (uint64)e->expr.get();
      });

  py::class_<ExprGroup>(m, "ExprGroup")
      .def(py::init<>())
      .def("size", [](ExprGroup *eg) { return eg->exprs.size(); })
      .def("push_back", &ExprGroup::push_back);

  py::class_<Stmt>(m, "Stmt");  // NOLINT(bugprone-unused-raii)

  m.def("insert_internal_func_call", [&](Operation *op, const ExprGroup &args) {
    return Expr::make<InternalFuncCallExpression>(op, args.exprs);
  });

  m.def("make_get_element_expr",
        Expr::make<GetElementExpression, const Expr &, std::vector<int>>);

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

  m.def("expr_atomic_mul", [&](const Expr &a, const Expr &b) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::mul, a, b);
  });

  m.def("expr_assume_in_range", assume_range);

  m.def("expr_loop_unique", loop_unique);

  m.def("expr_field", expr_field);

  m.def("expr_matrix_field", expr_matrix_field);

#define DEFINE_EXPRESSION_OP(x) m.def("expr_" #x, expr_##x);

  DEFINE_EXPRESSION_OP(neg)
  DEFINE_EXPRESSION_OP(sqrt)
  DEFINE_EXPRESSION_OP(round)
  DEFINE_EXPRESSION_OP(floor)
  DEFINE_EXPRESSION_OP(frexp)
  DEFINE_EXPRESSION_OP(ceil)
  DEFINE_EXPRESSION_OP(abs)
  DEFINE_EXPRESSION_OP(sin)
  DEFINE_EXPRESSION_OP(asin)
  DEFINE_EXPRESSION_OP(cos)
  DEFINE_EXPRESSION_OP(acos)
  DEFINE_EXPRESSION_OP(tan)
  DEFINE_EXPRESSION_OP(tanh)
  DEFINE_EXPRESSION_OP(inv)
  DEFINE_EXPRESSION_OP(rcp)
  DEFINE_EXPRESSION_OP(rsqrt)
  DEFINE_EXPRESSION_OP(exp)
  DEFINE_EXPRESSION_OP(log)
  DEFINE_EXPRESSION_OP(popcnt)

  DEFINE_EXPRESSION_OP(select)
  DEFINE_EXPRESSION_OP(ifte)

  DEFINE_EXPRESSION_OP(cmp_le)
  DEFINE_EXPRESSION_OP(cmp_lt)
  DEFINE_EXPRESSION_OP(cmp_ge)
  DEFINE_EXPRESSION_OP(cmp_gt)
  DEFINE_EXPRESSION_OP(cmp_ne)
  DEFINE_EXPRESSION_OP(cmp_eq)

  DEFINE_EXPRESSION_OP(bit_and)
  DEFINE_EXPRESSION_OP(bit_or)
  DEFINE_EXPRESSION_OP(bit_xor)
  DEFINE_EXPRESSION_OP(bit_shl)
  DEFINE_EXPRESSION_OP(bit_shr)
  DEFINE_EXPRESSION_OP(bit_sar)
  DEFINE_EXPRESSION_OP(bit_not)

  DEFINE_EXPRESSION_OP(logic_not)
  DEFINE_EXPRESSION_OP(logical_and)
  DEFINE_EXPRESSION_OP(logical_or)

  DEFINE_EXPRESSION_OP(add)
  DEFINE_EXPRESSION_OP(sub)
  DEFINE_EXPRESSION_OP(mul)
  DEFINE_EXPRESSION_OP(div)
  DEFINE_EXPRESSION_OP(truediv)
  DEFINE_EXPRESSION_OP(floordiv)
  DEFINE_EXPRESSION_OP(mod)
  DEFINE_EXPRESSION_OP(max)
  DEFINE_EXPRESSION_OP(min)
  DEFINE_EXPRESSION_OP(atan2)
  DEFINE_EXPRESSION_OP(pow)

#undef DEFINE_EXPRESSION_OP

  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>);
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>);
  m.def("make_frontend_assign_stmt",
        Stmt::make<FrontendAssignStmt, const Expr &, const Expr &>);

  m.def("make_arg_load_expr",
        Expr::make<ArgLoadExpression, int, const DataType &, bool, bool>,
        "arg_id"_a, "dt"_a, "is_ptr"_a = false, "create_load"_a = true);

  m.def("make_reference", Expr::make<ReferenceExpression, const Expr &>);

  m.def("make_external_tensor_expr",
        Expr::make<ExternalTensorExpression, const DataType &, int, int, int,
                   const std::vector<int> &, bool>);

  m.def("make_rand_expr", Expr::make<RandExpression, const DataType &>);

  m.def("make_const_expr_int",
        Expr::make<ConstExpression, const DataType &, int64>);

  m.def("make_const_expr_fp",
        Expr::make<ConstExpression, const DataType &, float64>);

  m.def("make_texture_ptr_expr", Expr::make<TexturePtrExpression, int, int>);
  m.def("make_rw_texture_ptr_expr",
        Expr::make<TexturePtrExpression, int, int, const BufferFormat &, int>);

  auto &&texture =
      py::enum_<TextureOpType>(m, "TextureOpType", py::arithmetic());
  for (int t = 0; t <= (int)TextureOpType::kStore; t++)
    texture.value(texture_op_type_name(TextureOpType(t)).c_str(),
                  TextureOpType(t));
  texture.export_values();

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
  m.def("is_quant", is_quant);
  m.def("is_integral", is_integral);
  m.def("is_signed", is_signed);
  m.def("is_real", is_real);
  m.def("is_unsigned", is_unsigned);
  m.def("is_tensor", is_tensor);

  m.def("data_type_name", data_type_name);

  m.def(
      "subscript_with_multiple_indices",
      Expr::make<IndexExpression, const Expr &, const std::vector<ExprGroup> &,
                 const std::vector<int> &, std::string>);

  m.def("get_external_tensor_element_dim", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->element_dim;
  });

  m.def("get_external_tensor_needs_grad", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->needs_grad;
  });

  m.def("get_external_tensor_element_shape", [](const Expr &expr) {
    TI_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt.get_shape();
  });

  m.def("get_external_tensor_dim", [](const Expr &expr) {
    if (expr.is<ExternalTensorExpression>()) {
      return expr.cast<ExternalTensorExpression>()->dim;
    } else if (expr.is<TexturePtrExpression>()) {
      return expr.cast<TexturePtrExpression>()->num_dims;
    } else {
      TI_ASSERT(false);
      return 0;
    }
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

  m.def("test_throw", [] { throw IRModified(); });

#if TI_WITH_LLVM
  m.def("libdevice_path", libdevice_path);
#endif

  m.def("host_arch", host_arch);
  m.def("arch_uses_llvm", arch_uses_llvm);

  m.def("set_lib_dir", [&](const std::string &dir) { compiled_lib_dir = dir; });
  m.def("set_tmp_dir", [&](const std::string &dir) { runtime_tmp_dir = dir; });

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
  m.def("get_llvm_target_support", [] {
#if defined(TI_WITH_LLVM)
    return LLVM_VERSION_STRING;
#else
    return "targets unsupported";
#endif
  });
  m.def("test_printf", [] { printf("test_printf\n"); });
  m.def("test_logging", [] { TI_INFO("test_logging"); });
  m.def("trigger_crash", [] { *(int *)(1) = 0; });
  m.def("get_max_num_indices", [] { return taichi_max_num_indices; });
  m.def("get_max_num_args", [] { return taichi_max_num_args; });
  m.def("test_threading", test_threading);
  m.def("is_extension_supported", is_extension_supported);

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

  m.def("promoted_type", promoted_type);

  // Note that it is important to specify py::return_value_policy::reference for
  // the factory methods, otherwise pybind11 will delete the Types owned by
  // TypeFactory on Python-scope pointer destruction.
  py::class_<TypeFactory>(m, "TypeFactory")
      .def("get_quant_int_type", &TypeFactory::get_quant_int_type,
           py::arg("num_bits"), py::arg("is_signed"), py::arg("compute_type"),
           py::return_value_policy::reference)
      .def("get_quant_fixed_type", &TypeFactory::get_quant_fixed_type,
           py::arg("digits_type"), py::arg("compute_type"), py::arg("scale"),
           py::return_value_policy::reference)
      .def("get_quant_float_type", &TypeFactory::get_quant_float_type,
           py::arg("digits_type"), py::arg("exponent_type"),
           py::arg("compute_type"), py::return_value_policy::reference)
      .def(
          "get_tensor_type",
          [&](TypeFactory *factory, std::vector<int> shape,
              const DataType &element_type) {
            return factory->create_tensor_type(shape, element_type);
          },
          py::return_value_policy::reference)
      .def(
          "get_struct_type",
          [&](TypeFactory *factory,
              std::vector<std::pair<DataType, std::string>> elements) {
            std::vector<StructMember> members;
            for (auto &[type, name] : elements) {
              members.push_back({type, name});
            }
            return DataType(factory->get_struct_type(members));
          },
          py::return_value_policy::reference);

  m.def("get_type_factory_instance", TypeFactory::get_instance,
        py::return_value_policy::reference);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<BitStructType>(m, "BitStructType");
  py::class_<BitStructTypeBuilder>(m, "BitStructTypeBuilder")
      .def(py::init<int>())
      .def("begin_placing_shared_exponent",
           &BitStructTypeBuilder::begin_placing_shared_exponent)
      .def("end_placing_shared_exponent",
           &BitStructTypeBuilder::end_placing_shared_exponent)
      .def("add_member", &BitStructTypeBuilder::add_member)
      .def("build", &BitStructTypeBuilder::build,
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

  // Sparse Matrix
  py::class_<SparseMatrixBuilder>(m, "SparseMatrixBuilder")
      .def(py::init<int, int, int, DataType, const std::string &, Program *>(),
           py::arg("rows"), py::arg("cols"), py::arg("max_num_triplets"),
           py::arg("dt") = PrimitiveType::f32,
           py::arg("storage_format") = "col_major", py::arg("prog") = nullptr)
      .def("print_triplets_eigen", &SparseMatrixBuilder::print_triplets_eigen)
      .def("print_triplets_cuda", &SparseMatrixBuilder::print_triplets_cuda)
      .def("get_ndarray_data_ptr", &SparseMatrixBuilder::get_ndarray_data_ptr)
      .def("build", &SparseMatrixBuilder::build)
      .def("build_cuda", &SparseMatrixBuilder::build_cuda)
      .def("get_addr", [](SparseMatrixBuilder *mat) { return uint64(mat); });

  py::class_<SparseMatrix>(m, "SparseMatrix")
      .def(py::init<>())
      .def(py::init<int, int, DataType>(), py::arg("rows"), py::arg("cols"),
           py::arg("dt") = PrimitiveType::f32)
      .def(py::init<SparseMatrix &>())
      .def("to_string", &SparseMatrix::to_string)
      .def("get_element", &SparseMatrix::get_element<float32>)
      .def("set_element", &SparseMatrix::set_element<float32>)
      .def("mmwrite", &SparseMatrix::mmwrite)
      .def("num_rows", &SparseMatrix::num_rows)
      .def("num_cols", &SparseMatrix::num_cols)
      .def("get_data_type", &SparseMatrix::get_data_type);

#define MAKE_SPARSE_MATRIX(TYPE, STORAGE, VTYPE)                             \
  using STORAGE##TYPE##EigenMatrix =                                         \
      Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;                      \
  py::class_<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>, SparseMatrix>(   \
      m, #VTYPE #STORAGE "_EigenSparseMatrix")                               \
      .def(py::init<int, int, DataType>())                                   \
      .def(py::init<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix> &>())      \
      .def(py::init<const STORAGE##TYPE##EigenMatrix &>())                   \
      .def(py::self += py::self)                                             \
      .def(py::self + py::self)                                              \
      .def(py::self -= py::self)                                             \
      .def(py::self - py::self)                                              \
      .def(py::self *= float##TYPE())                                        \
      .def(py::self *float##TYPE())                                          \
      .def(float##TYPE() * py::self)                                         \
      .def(py::self *py::self)                                               \
      .def("matmul", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::matmul) \
      .def("spmv", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::spmv)     \
      .def("transpose",                                                      \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::transpose)        \
      .def("get_element",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::get_element<      \
               float##TYPE>)                                                 \
      .def("set_element",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::set_element<      \
               float##TYPE>)                                                 \
      .def("mat_vec_mul",                                                    \
           &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::mat_vec_mul<      \
               Eigen::VectorX##VTYPE>);

  MAKE_SPARSE_MATRIX(32, ColMajor, f);
  MAKE_SPARSE_MATRIX(32, RowMajor, f);
  MAKE_SPARSE_MATRIX(64, ColMajor, d);
  MAKE_SPARSE_MATRIX(64, RowMajor, d);

  py::class_<CuSparseMatrix, SparseMatrix>(m, "CuSparseMatrix")
      .def(py::init<int, int, DataType>())
      .def(py::init<const CuSparseMatrix &>())
      .def("spmv", &CuSparseMatrix::nd_spmv)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * float32())
      .def(float32() * py::self)
      .def("matmul", &CuSparseMatrix::matmul)
      .def("transpose", &CuSparseMatrix::transpose)
      .def("get_element", &CuSparseMatrix::get_element)
      .def("to_string", &CuSparseMatrix::to_string);

  py::class_<SparseSolver>(m, "SparseSolver")
      .def("compute", &SparseSolver::compute)
      .def("analyze_pattern", &SparseSolver::analyze_pattern)
      .def("factorize", &SparseSolver::factorize)
      .def("info", &SparseSolver::info);

#define REGISTER_EIGEN_SOLVER(dt, type, order, fd)                           \
  py::class_<EigenSparseSolver##dt##type##order, SparseSolver>(              \
      m, "EigenSparseSolver" #dt #type #order)                               \
      .def("compute", &EigenSparseSolver##dt##type##order::compute)          \
      .def("analyze_pattern",                                                \
           &EigenSparseSolver##dt##type##order::analyze_pattern)             \
      .def("factorize", &EigenSparseSolver##dt##type##order::factorize)      \
      .def("solve",                                                          \
           &EigenSparseSolver##dt##type##order::solve<Eigen::VectorX##fd>)   \
      .def("solve_rf",                                                       \
           &EigenSparseSolver##dt##type##order::solve_rf<Eigen::VectorX##fd, \
                                                         dt>)                \
      .def("info", &EigenSparseSolver##dt##type##order::info);

  REGISTER_EIGEN_SOLVER(float32, LLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float64, LLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, COLAMD, d)

  py::class_<CuSparseSolver, SparseSolver>(m, "CuSparseSolver")
      .def("compute", &CuSparseSolver::compute)
      .def("analyze_pattern", &CuSparseSolver::analyze_pattern)
      .def("factorize", &CuSparseSolver::factorize)
      .def("solve_rf", &CuSparseSolver::solve_rf)
      .def("info", &CuSparseSolver::info);

  m.def("make_sparse_solver", &make_sparse_solver);
  m.def("make_cusparse_solver", &make_cusparse_solver);

  // Conjugate Gradient solver
  py::class_<CG<Eigen::VectorXf, float>>(m, "CGf")
      .def(py::init<SparseMatrix &, int, float, bool>())
      .def("solve", &CG<Eigen::VectorXf, float>::solve)
      .def("set_x", &CG<Eigen::VectorXf, float>::set_x)
      .def("get_x", &CG<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &CG<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray", &CG<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success", &CG<Eigen::VectorXf, float>::is_success);
  py::class_<CG<Eigen::VectorXd, double>>(m, "CGd")
      .def(py::init<SparseMatrix &, int, double, bool>())
      .def("solve", &CG<Eigen::VectorXd, double>::solve)
      .def("set_x", &CG<Eigen::VectorXd, double>::set_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXd, double>::set_x_ndarray)
      .def("get_x", &CG<Eigen::VectorXd, double>::get_x)
      .def("set_b_ndarray", &CG<Eigen::VectorXd, double>::set_b_ndarray)
      .def("set_b", &CG<Eigen::VectorXd, double>::set_b)
      .def("is_success", &CG<Eigen::VectorXd, double>::is_success);
  m.def("make_float_cg_solver", [](SparseMatrix &A, int max_iters, float tol,
                                   bool verbose) {
    return make_cg_solver<Eigen::VectorXf, float>(A, max_iters, tol, verbose);
  });
  m.def("make_double_cg_solver", [](SparseMatrix &A, int max_iters, float tol,
                                    bool verbose) {
    return make_cg_solver<Eigen::VectorXd, double>(A, max_iters, tol, verbose);
  });

  py::class_<CUCG>(m, "CUCG").def("solve", &CUCG::solve);
  m.def("make_cucg_solver", make_cucg_solver);

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

  py::class_<mesh::Mesh>(m, "Mesh");        // NOLINT(bugprone-unused-raii)
  py::class_<mesh::MeshPtr>(m, "MeshPtr");  // NOLINT(bugprone-unused-raii)

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
           SNode *patch_offset, SNode *offset) {
          mesh_ptr.ptr->relations.insert(std::pair(
              type, mesh::MeshLocalRelation(value, patch_offset, offset)));
        });

  m.def("wait_for_debugger", []() {
#ifdef WIN32
    while (!::IsDebuggerPresent())
      ::Sleep(100);
#endif
  });

  auto operationClass = py::class_<Operation>(m, "Operation");
  auto internalOpClass = py::class_<InternalOp>(m, "InternalOp");

#define PER_INTERNAL_OP(x)                                           \
  internalOpClass.def_property_readonly_static(                      \
      #x, [](py::object) { return Operations::get(InternalOp::x); }, \
      py::return_value_policy::reference);
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
}

}  // namespace taichi
