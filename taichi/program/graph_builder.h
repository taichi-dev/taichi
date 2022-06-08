#pragma once

#include <string>
#include <vector>

#include "taichi/ir/type.h"
#include "taichi/aot/graph_data.h"

namespace taichi {
namespace lang {
class Kernel;
class GraphBuilder;

class Node {
 public:
  Node() = default;
  virtual ~Node() = default;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(Node &&) = default;

  virtual void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) = 0;
};

class Dispatch : public Node {
 public:
  explicit Dispatch(Kernel *kernel, const std::vector<aot::Arg> &args)
      : kernel_(kernel), symbolic_args_(args) {
  }

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

 private:
  mutable bool serialized_{false};
  Kernel *kernel_{nullptr};
  std::vector<aot::Arg> symbolic_args_;
};

class Sequential : public Node {
 public:
  explicit Sequential(GraphBuilder *graph) : owning_graph_(graph) {
  }

  void append(Node *node);

  void dispatch(Kernel *kernel, const std::vector<aot::Arg> &args);

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

 private:
  std::vector<Node *> sequence_;
  GraphBuilder *owning_graph_{nullptr};
};

class GraphBuilder {
 public:
  explicit GraphBuilder();

  // TODO: compile() can take in Arch argument
  std::unique_ptr<aot::CompiledGraph> compile();

  Node *new_dispatch_node(Kernel *kernel, const std::vector<aot::Arg> &args);

  Sequential *new_sequential_node();

  void dispatch(Kernel *kernel, const std::vector<aot::Arg> &args);

  Sequential *seq() const;

 private:
  std::unique_ptr<Sequential> seq_{nullptr};
  std::vector<std::unique_ptr<Node>> all_nodes_;
};

}  // namespace lang
}  // namespace taichi
