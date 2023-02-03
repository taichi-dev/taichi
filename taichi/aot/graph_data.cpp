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
    RuntimeContext ctx = ctx_;

    TI_ASSERT(dispatch.ti_kernel || dispatch.compiled_kernel);

    // Populate args metadata into RuntimeContext
    const auto &symbolic_args_ = dispatch.symbolic_args;
    for (int i = 0; i < symbolic_args_.size(); ++i) {
      auto &symbolic_arg = symbolic_args_[i];
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
                    symbolic_arg.name, symbolic_arg.field_dim,
                    arr->shape.size());

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
        ctx.set_arg_ndarray(i, arr->get_device_allocation_ptr_as_int(),
                            arr->shape);
      } else if (symbolic_arg.tag == aot::ArgKind::kScalar ||
                 symbolic_arg.tag == aot::ArgKind::kMatrix) {
        TI_ASSERT(ival.tag == aot::ArgKind::kScalar);
        // Matrix args are flattened so they're same as scalars.
        ctx.set_arg(i, ival.val);
      } else if (symbolic_arg.tag == aot::ArgKind::kTexture) {
        TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
        Texture *tex = reinterpret_cast<Texture *>(ival.val);
        ctx.set_arg_texture(i, tex->get_device_allocation_ptr_as_int());
      } else if (symbolic_arg.tag == aot::ArgKind::kRWTexture) {
        TI_ASSERT(ival.tag == aot::ArgKind::kTexture);
        Texture *tex = reinterpret_cast<Texture *>(ival.val);
        ctx.set_arg_rw_texture(i, tex->get_device_allocation_ptr_as_int(),
                               tex->get_size());
      } else {
        TI_ERROR("Error in compiled graph: unknown tag {}", ival.tag);
      }
    }

    if (dispatch.compiled_kernel) {
      // Run cgraph loaded from AOT module
      dispatch.compiled_kernel->launch(&ctx);
    } else {
      // JIT & Run
      TI_ASSERT(dispatch.ti_kernel);
      lang::Kernel::LaunchContextBuilder launch_ctx(dispatch.ti_kernel, &ctx);
      auto *ker = dispatch.ti_kernel;
      ker->operator()(ker->program->compile_config(), launch_ctx);
    }
  }
}
}  // namespace aot
}  // namespace taichi::lang
