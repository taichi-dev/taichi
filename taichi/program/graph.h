#pragma once

#include <string>
#include <vector>
#include <unordered_set>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/ir/type.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_builder.h"

namespace taichi {
namespace lang {
class Kernel;
class Graph;

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
  std::unique_ptr<aot::Kernel> compiled_kernel_{nullptr};
  std::vector<aot::Arg> symbolic_args_;
};

class Sequential : public Node {
 public:
  explicit Sequential(Graph *graph) : owning_graph_(graph) {
  }

  void append(Node *node);

  void dispatch(Kernel *kernel, const std::vector<aot::Arg> &args);

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override;

 private:
  std::vector<Node *> sequence_;
  Graph *owning_graph_{nullptr};
};

/*
 * Graph class works as both builder and runner.
 *
 * Two typical workflows using Graph:
 * - build graph -> compile -> run
 * - build graph -> compile -> serialize -> deserialize -> run
 *
 * Thus Graph can be constructed in two ways, either as an empty object
 * or from an `aot::CompiledGraph` loaded from aot module.
 *
 * Currently Graph only supports sequential launches without returning value
 * to host.
 */
class Graph {
 public:
  explicit Graph(std::string name);

  explicit Graph(std::string name, const aot::CompiledGraph &compiled)
      : name_(name), compiled_graph_(compiled) {
  }

  // TODO: compile() can take in Arch argument
  void compile();

  void run(const std::unordered_map<std::string, aot::IValue> &args) const;

  Node *new_dispatch_node(Kernel *kernel, const std::vector<aot::Arg> &args);

  Sequential *new_sequential_node();

  void dispatch(Kernel *kernel, const std::vector<aot::Arg> &args);

  Sequential *seq() const;

  aot::CompiledGraph compiled_graph() const {
    return compiled_graph_;
  }

  std::string name() const {
    return name_;
  }

 private:
  std::string name_;
  std::unique_ptr<Sequential> seq_{nullptr};
  std::vector<std::unique_ptr<Node>> all_nodes_;
  aot::CompiledGraph compiled_graph_;
};

}  // namespace lang
}  // namespace taichi
