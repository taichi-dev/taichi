#include <regex>
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
      size_t mat_arg_count = 0;
      if (found == args.end()) {
        // Try find matrix args
        // TODO replace this logic when C++ matrix is available.
        std::regex mat_arg_regex(symbolic_arg.name + "_mat_.[0-9]*");
        std::smatch arg_match;
        mat_arg_count = 0;
        for (auto arg : args) {
          if (std::regex_match(arg.first, arg_match, mat_arg_regex)) {
            auto find_idx = arg_match.str().find_last_of("mat_");
            std::string idx_str(arg_match.str().begin() + find_idx + 1,
                                arg_match.str().end());
            size_t mat_arg_idx = std::stoi(idx_str);
            ctx.set_arg(argument_slot_id + mat_arg_idx, arg.second.val);
            mat_arg_count++;
          }
        }
        argument_slot_id += mat_arg_count;
        if (mat_arg_count > 0) {
          continue;
        }
      }
      TI_ERROR_IF(found == args.end() && mat_arg_count == 0,
                  "Missing runtime value for {}", symbolic_arg.name);
      const aot::IValue &ival = found->second;
      if (ival.tag == aot::ArgKind::kNdarray) {
        Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
        TI_ERROR_IF(arr->element_shape != symbolic_arg.element_shape,
                    "Mismatched shape information for argument {}",
                    symbolic_arg.name);
        set_runtime_ctx_ndarray(&ctx, argument_slot_id++, arr);
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
