#include "taichi/aot/graph_data.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"

#include <numeric>

namespace taichi {
namespace lang {
namespace aot {

void CompiledGraph::run(
    const std::unordered_map<std::string, IValue> &args) const {
  for (const auto &dispatch : dispatches) {
    RuntimeContext ctx = ctx_;

    TI_ASSERT(dispatch.compiled_kernel);

    // Populate args metadata into RuntimeContext
    const auto &symbolic_args_ = dispatch.symbolic_args;
    for (int i = 0; i < symbolic_args_.size(); ++i) {
      auto &symbolic_arg = symbolic_args_[i];
      auto found = args.find(symbolic_arg.name);
      TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                  symbolic_arg.name);
      const aot::IValue &ival = found->second;
      if (ival.tag == aot::ArgKind::kNdarray) {
        Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
        TI_ERROR_IF(arr->element_shape != symbolic_arg.element_shape,
                    "Mismatched shape information for argument {}",
                    symbolic_arg.name);
        TI_ERROR_IF(arr->shape.size() != symbolic_arg.field_dim,
                    "Dispatch node is compiled for argument {} with "
                    "field_dim={} but got an ndarray with field_dim={}",
                    symbolic_arg.name, symbolic_arg.field_dim,
                    arr->shape.size());
        TI_ERROR_IF(arr->dtype != symbolic_arg.dtype(),
                    "Dispatch node is compiled for argument {} with "
                    "dtype={} but got an ndarray with dtype={}",
                    symbolic_arg.name, symbolic_arg.dtype().to_string(),
                    arr->dtype.to_string());

        ctx.set_arg_ndarray(i, arr->get_device_allocation_ptr_as_int(),
                            arr->shape);
      } else if (ival.tag == aot::ArgKind::kScalar) {
        ctx.set_arg(i, ival.val);
      } else if (ival.tag == aot::ArgKind::kTexture) {
        Texture *tex = reinterpret_cast<Texture *>(ival.val);
        ctx.set_arg_texture(i, tex->get_device_allocation_ptr_as_int());
      } else if (ival.tag == aot::ArgKind::kRWTexture) {
        Texture *tex = reinterpret_cast<Texture *>(ival.val);
        ctx.set_arg_rw_texture(i, tex->get_device_allocation_ptr_as_int());
      } else {
        TI_ERROR("Error in compiled graph: unknown tag {}", ival.tag);
      }
    }
    dispatch.compiled_kernel->launch(&ctx);
  }
}
}  // namespace aot
}  // namespace lang
}  // namespace taichi
