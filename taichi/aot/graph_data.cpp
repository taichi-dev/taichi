#include "taichi/aot/graph_data.h"
#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/program/kernel.h"

#include <numeric>

namespace taichi::lang {
namespace aot {

void CompiledGraph::run(
    const std::unordered_map<std::string, IValue> &args) const {
  for (const auto &dispatch : dispatches) {
    TI_ASSERT(dispatch.compiled_kernel);
    LaunchContextBuilder launch_ctx(dispatch.compiled_kernel);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    // Run cgraph loaded from AOT module
    dispatch.compiled_kernel->launch(launch_ctx);
  }
}

void CompiledGraph::jit_run(
    const CompileConfig &compile_config,
    const std::unordered_map<std::string, IValue> &args) const {
  for (const auto &dispatch : dispatches) {
    TI_ASSERT(dispatch.ti_kernel);
    LaunchContextBuilder launch_ctx(dispatch.ti_kernel);
    init_runtime_context(dispatch.symbolic_args, args, launch_ctx);
    // Compile & Run (JIT): The compilation result will be cached, so don't
    // worry that the kernels dispatched by this cgraph will be compiled
    // repeatedly.
    auto *prog = dispatch.ti_kernel->program;
    const auto &compiled_kernel_data = prog->compile_kernel(
        compile_config, prog->get_device_caps(), *dispatch.ti_kernel);
    prog->launch_kernel(compiled_kernel_data, launch_ctx);
  }
}

// static
void CompiledGraph::init_runtime_context(
    const std::vector<Arg> &paramter_list,
    const std::unordered_map<std::string, IValue> &args,
    LaunchContextBuilder &ctx) {
  for (int i = 0; i < paramter_list.size(); ++i) {
    auto &symbolic_arg = paramter_list[i];
    if (symbolic_arg.tag == aot::ArgKind::kMatrix) {
      int size = symbolic_arg.element_shape[0] * symbolic_arg.element_shape[1];
      for (int j = 0; j < size; j++) {
        auto found = args.find(symbolic_arg.name + "_" + std::to_string(j));
        TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                    symbolic_arg.name);
        const aot::IValue &ival = found->second;
        TI_ASSERT(ival.tag == aot::ArgKind::kScalar);
        int type_size = data_type_size(symbolic_arg.dtype());
        switch (type_size) {
          case 1:
            ctx.set_struct_arg_impl(
                {i, j}, taichi_union_cast_with_different_sizes<int8>(ival.val));
            break;
          case 2:
            ctx.set_struct_arg_impl(
                {i, j},
                taichi_union_cast_with_different_sizes<int16>(ival.val));
            break;
          case 4:
            ctx.set_struct_arg_impl(
                {i, j},
                taichi_union_cast_with_different_sizes<int32>(ival.val));
            break;
          case 8:
            ctx.set_struct_arg_impl(
                {i, j},
                taichi_union_cast_with_different_sizes<int64>(ival.val));
            break;
          default:
            TI_ERROR("Unsupported type size {}", type_size);
        }
      }
      continue;
    }
    auto found = args.find(symbolic_arg.name);
    TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                symbolic_arg.name);
    const aot::IValue &ival = found->second;
    if (symbolic_arg.tag == aot::ArgKind::kNdarray) {
      TI_ASSERT(ival.tag == aot::ArgKind::kNdarray);
      Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);

      TI_ERROR_IF(arr->get_element_shape() != symbolic_arg.element_shape,
                  "Mismatched shape information for argument {}",
                  symbolic_arg.name);
      TI_ERROR_IF(arr->shape.size() != symbolic_arg.field_dim,
                  "Dispatch node is compiled for argument {} with "
                  "field_dim={} but got an ndarray with field_dim={}",
                  symbolic_arg.name, symbolic_arg.field_dim, arr->shape.size());

      // CGraph uses aot::Arg as symbolic argument, which represents
      // TensorType via combination of element_shape and PrimitiveTypeID
      // Therefore we only check for element_type for now.
      //
      // TODO(zhanlue): Replace all "element_shape + PrimitiveType" use cases
      // with direct use of "TensorType",
      //                In the end, "element_shape" should only appear inside
      //                TensorType and nowhere else.
      //
      //                This refactor includes aot::Arg, kernel::Arg,
      //                MetalDataType, and more...
      DataType symbolic_arg_primitive_dtype = symbolic_arg.dtype();
      if (symbolic_arg.dtype()->is<TensorType>()) {
        symbolic_arg_primitive_dtype =
            symbolic_arg.dtype()->cast<TensorType>()->get_element_type();
      }

      DataType arr_primitive_dtype = arr->dtype;
      if (arr->dtype->is<TensorType>()) {
        arr_primitive_dtype =
            arr->dtype->cast<TensorType>()->get_element_type();
      }

      TI_ERROR_IF(arr_primitive_dtype != symbolic_arg_primitive_dtype,
                  "Dispatch node is compiled for argument {} with "
                  "dtype={} but got an ndarray with dtype={}",
                  symbolic_arg.name, symbolic_arg_primitive_dtype.to_string(),
                  arr_primitive_dtype.to_string());
      ctx.set_arg_ndarray(i, *arr);
    } else if (symbolic_arg.tag == aot::ArgKind::kScalar) {
      TI_ASSERT(ival.tag == aot::ArgKind::kScalar);
      // Matrix args are flattened so they're same as scalars.
      int type_size = data_type_size(symbolic_arg.dtype());
      switch (type_size) {
        case 1:
          ctx.set_arg(i,
                      taichi_union_cast_with_different_sizes<int8>(ival.val));
          break;
        case 2:
          ctx.set_arg(i,
                      taichi_union_cast_with_different_sizes<int16>(ival.val));
          break;
        case 4:
          ctx.set_arg(i,
                      taichi_union_cast_with_different_sizes<int32>(ival.val));
          break;
        case 8:
          ctx.set_arg(i,
                      taichi_union_cast_with_different_sizes<int64>(ival.val));
          break;
        default:
          TI_ERROR("Unsupported type size {}", type_size);
      }
    } else if (symbolic_arg.tag == aot::ArgKind::kTexture) {
      TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
      Texture *tex = reinterpret_cast<Texture *>(ival.val);
      ctx.set_arg_texture(i, *tex);
    } else if (symbolic_arg.tag == aot::ArgKind::kRWTexture) {
      TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
      Texture *tex = reinterpret_cast<Texture *>(ival.val);
      ctx.set_arg_rw_texture(i, *tex);
    } else {
      TI_ERROR("Error in compiled graph: unknown tag {}", ival.tag);
    }
  }
}

}  // namespace aot
}  // namespace taichi::lang
