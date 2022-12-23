#include "taichi/program/graph_builder.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi::lang {
void Dispatch::compile(
    std::vector<aot::CompiledDispatch> &compiled_dispatches) {
  aot::CompiledDispatch dispatch;
  dispatch.kernel_name = kernel_->get_name();
  dispatch.symbolic_args = symbolic_args_;
  dispatch.ti_kernel = kernel_;
  dispatch.compiled_kernel = nullptr;
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

void Sequential::dispatch(Kernel *kernel, const std::vector<aot::Arg> &args) {
  Node *n = owning_graph_->new_dispatch_node(kernel, args);
  sequence_.push_back(n);
}

GraphBuilder::GraphBuilder() {
  seq_ = std::make_unique<Sequential>(this);
}

Node *GraphBuilder::new_dispatch_node(Kernel *kernel,
                                      const std::vector<aot::Arg> &args) {
  for (const auto &arg : args) {
    if (all_args_.find(arg.name) != all_args_.end()) {
      TI_ERROR_IF(all_args_[arg.name] != arg,
                  "An arg with name {} already exists!", arg.name);
    } else {
      all_args_[arg.name] = arg;
    }
  }
  all_nodes_.push_back(std::make_unique<Dispatch>(kernel, args));
  return all_nodes_.back().get();
}

Sequential *GraphBuilder::new_sequential_node() {
  all_nodes_.push_back(std::make_unique<Sequential>(this));
  return static_cast<Sequential *>(all_nodes_.back().get());
}

std::unique_ptr<aot::CompiledGraph> GraphBuilder::compile() {
  std::vector<aot::CompiledDispatch> dispatches;
  seq()->compile(dispatches);
  aot::CompiledGraph graph{dispatches, all_args_};
  return std::make_unique<aot::CompiledGraph>(std::move(graph));
}

Sequential *GraphBuilder::seq() const {
  return seq_.get();
}

void GraphBuilder::dispatch(Kernel *kernel, const std::vector<aot::Arg> &args) {
  seq()->dispatch(kernel, args);
}

}  // namespace taichi::lang
