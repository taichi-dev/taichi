#include "taichi/program/graph.h"
#include "taichi/program/kernel.h"
#include "taichi/aot/module_builder.h"
#include "spdlog/fmt/fmt.h"

#include <fstream>

namespace taichi {
namespace lang {

void Dispatch::compile(
    std::vector<aot::CompiledDispatch> &compiled_dispatches) {
  compiled_kernel_ = kernel_->compile_to_aot_kernel();
  aot::CompiledDispatch dispatch{kernel_->get_name(), symbolic_args_,
                                 compiled_kernel_.get()};
  compiled_dispatches.push_back(std::move(dispatch));
}

void Sequential::compile(
    std::vector<aot::CompiledDispatch> &compiled_dispatches) {
  // In the future we can do more across-kernel optimization here.
  for (Node *n : sequence_) {
    n->compile(compiled_dispatches);
  }
}

void Sequential::append(Node *node) {
  sequence_.push_back(node);
}

void Sequential::emplace(Kernel *kernel, const std::vector<aot::Arg> &args) {
  Node *n = owning_graph_->create_dispatch(kernel, args);
  sequence_.push_back(n);
}

Graph::Graph(std::string name) : name_(name) {
  seq_ = std::make_unique<Sequential>(this);
}
Node *Graph::create_dispatch(Kernel *kernel,
                             const std::vector<aot::Arg> &args) {
  all_nodes_.push_back(std::make_unique<Dispatch>(kernel, args));
  return all_nodes_.back().get();
}

Sequential *Graph::create_sequential() {
  all_nodes_.push_back(std::make_unique<Sequential>(this));
  return static_cast<Sequential *>(all_nodes_.back().get());
}

void Graph::compile() {
  seq()->compile(compiled_graph_.dispatches);
}

Sequential *Graph::seq() const {
  return seq_.get();
}

void Graph::emplace(Kernel *kernel, const std::vector<aot::Arg> &args) {
  seq()->emplace(kernel, args);
}

void Graph::run(
    const std::unordered_map<std::string, aot::IValue> &args) const {
  RuntimeContext ctx;
  for (const auto &dispatch : compiled_graph_.dispatches) {
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
        TI_ERROR_IF((symbolic_arg.tag != ival.tag) ||
                        (symbolic_arg.element_shape != arr->shape),
                    "Mismatched shape information for argument {}",
                    symbolic_arg.name);
        set_runtime_ctx_ndarray(&ctx, i, arr);
      } else {
        TI_ERROR_IF(symbolic_arg.tag != aot::ArgKind::SCALAR,
                    "Required a scalar for argument {}", symbolic_arg.name);
        ctx.set_arg(i, ival.val);
      }
    }

    dispatch.compiled_kernel->launch(&ctx);
  }
}
}  // namespace lang
}  // namespace taichi
