#include "taichi/aot/graph_data.h"
#include "taichi/program/ndarray.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace aot {
void CompiledGraph::run(
    const std::unordered_map<std::string, IValue> &args) const {
  RuntimeContext ctx;
  for (const auto &dispatch : dispatches) {
    memset(&ctx, 0, sizeof(RuntimeContext));

    TI_ASSERT(dispatch.compiled_kernel);
    // Populate args metadata into RuntimeContext
    const auto &symbolic_args_ = dispatch.symbolic_args;
    for (int i = 0; i < symbolic_args_.size(); ++i) {
      auto &symbolic_arg = symbolic_args_[i];
      auto found = args.find(symbolic_arg.name);
      TI_ERROR_IF(found == args.end(), "Missing runtime value for {}",
                  symbolic_arg.name);
      const aot::IValue &ival = found->second;
      if (ival.tag == aot::ArgKind::NDARRAY) {
        Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
        TI_ERROR_IF(ival.tag != aot::ArgKind::NDARRAY,
                    "Required a ndarray for argument {}", symbolic_arg.name);
        auto ndarray_elem_shape = std::vector<int>(
            arr->shape.end() - symbolic_arg.element_shape.size(),
            arr->shape.end());
        TI_ERROR_IF(ndarray_elem_shape != symbolic_arg.element_shape,
                    "Mismatched shape information for argument {}",
                    symbolic_arg.name);
        set_runtime_ctx_ndarray(&ctx, i, arr);
      } else {
        TI_ERROR_IF(ival.tag != aot::ArgKind::SCALAR,
                    "Required a scalar for argument {}", symbolic_arg.name);
        ctx.set_arg(i, ival.val);
      }
    }

    dispatch.compiled_kernel->launch(&ctx);
  }
}
}  // namespace aot
}  // namespace lang
}  // namespace taichi
