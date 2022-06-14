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
    int argument_slot_id = 0;
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
        set_runtime_ctx_ndarray(&ctx, argument_slot_id++, arr);
        } else if (ival.tag == aot::ArgKind::kMatrix) {
          auto mat_arr = reinterpret_cast<const std::vector<int>*>(ival.val);
          TI_WARN("MATRIX LEN {}", mat_arr->size())
          for (int k = 0; k < mat_arr->size(); ++k) {
            TI_WARN("SET ARG {}", mat_arr->at(k));
            ctx.set_arg(argument_slot_id++, val->at(k));
          }
        } else if (ival.tag == aot::ArgKind::kScalar) {
          ctx.set_arg(argument_slot_id++, ival.val);
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
