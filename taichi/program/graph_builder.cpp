#include "taichi/program/graph_builder.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {
void Dispatch::compile(
    std::vector<aot::CompiledDispatch> &compiled_dispatches) {
  if (kernel_->compiled_aot_kernel() == nullptr) {
    kernel_->compile_to_aot_kernel();
  }
  aot::CompiledDispatch dispatch{kernel_->get_name(), symbolic_args_,
                                 kernel_->compiled_aot_kernel()};
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
  aot::CompiledGraph graph{dispatches};
  return std::make_unique<aot::CompiledGraph>(std::move(graph));
}

Sequential *GraphBuilder::seq() const {
  return seq_.get();
}

void GraphBuilder::dispatch(Kernel *kernel, const std::vector<aot::Arg> &args) {
  seq()->dispatch(kernel, args);
}

}  // namespace lang
}  // namespace taichi
