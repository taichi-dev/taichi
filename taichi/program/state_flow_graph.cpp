#include "taichi/program/state_flow_graph.h"

#include "taichi/ir/transforms.h"
#include "taichi/program/async_engine.h"
#include "taichi/util/bit.h"

#include <sstream>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// TODO: rename state to edge since we have not only state flow edges but also
// dependency edges.

std::string StateFlowGraph::Node::string() const {
  return fmt::format("[node: {}:{}]", task_name, launch_id);
}

StateFlowGraph::StateFlowGraph(IRBank *ir_bank): ir_bank_(ir_bank) {
  nodes_.push_back(std::make_unique<Node>());
  initial_node_ = nodes_.back().get();
  initial_node_->task_name = "initial_state";
  initial_node_->launch_id = 0;
  initial_node_->is_initial_node = true;
}

void StateFlowGraph::clear() {
  // TODO: GC here?
  nodes_.resize(1);  // Erase all nodes except the initial one
  initial_node_->output_edges.clear();
  latest_state_owner_.clear();

  // Do not clear task_name_to_launch_ids_.
}

void StateFlowGraph::insert_task(const TaskLaunchRecord &rec,
                                 const TaskMeta &task_meta) {
  auto node = std::make_unique<Node>();
  node->rec = rec;
  node->task_name = task_meta.kernel_name;
  node->input_states = task_meta.input_states;
  node->output_states = task_meta.output_states;
  node->task_type = task_meta.type;
  {
    int &id = task_name_to_launch_ids_[node->task_name];
    node->launch_id = id;
    ++id;
  }
  for (auto input_state : task_meta.input_states) {
    if (latest_state_owner_.find(input_state) == latest_state_owner_.end()) {
      latest_state_owner_[input_state] = initial_node_;
    }
    insert_state_flow(latest_state_owner_[input_state], node.get(),
                      input_state);
  }
  for (auto output_state : task_meta.output_states) {
    latest_state_owner_[output_state] = node.get();
    if (latest_state_readers_.find(output_state) ==
        latest_state_readers_.end()) {
      latest_state_readers_[output_state].insert(initial_node_);
    }
    for (auto &d : latest_state_readers_[output_state]) {
      // insert a dependency edge
      insert_state_flow(d, node.get(), output_state);
    }
    latest_state_readers_[output_state].clear();
  }

  // Note that this loop must happen AFTER the previous one
  for (auto input_state : task_meta.input_states) {
    latest_state_readers_[input_state].insert(node.get());
  }
  nodes_.push_back(std::move(node));
}

void StateFlowGraph::insert_state_flow(Node *from, Node *to, AsyncState state) {
  TI_ASSERT(from != nullptr);
  TI_ASSERT(to != nullptr);
  from->output_edges[state].insert(to);
  to->input_edges[state].insert(from);
}

bool StateFlowGraph::fuse() {
  using SFGNode = StateFlowGraph::Node;
  using bit::Bitset;
  const int n = nodes_.size();
  for (int i = 0; i < n; i++) {
    nodes_[i]->node_id = i;
  }

  // Compute the transitive closure.
  auto has_path = std::make_unique<Bitset[]>(n);
  auto has_path_reverse = std::make_unique<Bitset[]>(n);
  // has_path[i][j] denotes if there is a path from i to j.
  // has_path_reverse[i][j] denotes if there is a path from j to i.
  for (int i = 0; i < n; i++) {
    has_path[i] = Bitset(n);
    has_path[i][i] = true;
    has_path_reverse[i] = Bitset(n);
    has_path_reverse[i][i] = true;
  }
  for (int i = 0; i < n; i++) {
    for (auto &edges : nodes_[i]->output_edges) {
      for (auto &edge : edges.second) {
        // Assume nodes are sorted in topological order.
        TI_ASSERT(edge->node_id > i);
        has_path[edge->node_id] |= has_path[i];
      }
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    for (auto &edges : nodes_[i]->input_edges) {
      for (auto &edge : edges.second) {
        TI_ASSERT(edge->node_id < i);
        has_path_reverse[edge->node_id] |= has_path_reverse[i];
      }
    }
  }

  // Cache the result that if each pair is fusable by task types.
  // TODO: improve this
  auto task_type_fusable = std::make_unique<Bitset[]>(n);
  // nodes_[0] is the initial node.
  for (int i = 1; i < n; i++) {
    auto &rec_i = nodes_[i]->rec;
    auto *task_i = rec_i.stmt();
    for (int j = i + 1; j < n; j++) {
      auto &rec_j = nodes_[j]->rec;
      auto *task_j = rec_j.stmt();
      bool is_same_struct_for =
          task_i->task_type == OffloadedStmt::struct_for &&
          task_j->task_type == OffloadedStmt::struct_for &&
          task_i->snode == task_j->snode &&
          task_i->block_dim == task_j->block_dim;
      // TODO: a few problems with the range-for test condition:
      // 1. This could incorrectly fuse two range-for kernels that have
      // different sizes, but then the loop ranges get padded to the same
      // power-of-two (E.g. maybe a side effect when a struct-for is demoted to
      // range-for).
      // 2. It has also fused range-fors that have the same linear range, but
      // are of different dimensions of loop indices, e.g. (16, ) and (4, 4).
      bool is_same_range_for = task_i->task_type == OffloadedStmt::range_for &&
                               task_j->task_type == OffloadedStmt::range_for &&
                               task_i->const_begin && task_j->const_begin &&
                               task_i->const_end && task_j->const_end &&
                               task_i->begin_value == task_j->begin_value &&
                               task_i->end_value == task_j->end_value;
      bool are_both_serial = task_i->task_type == OffloadedStmt::serial &&
                             task_j->task_type == OffloadedStmt::serial;
      const bool same_kernel = (rec_i.kernel == rec_j.kernel);
      bool kernel_args_match = true;
      if (!same_kernel) {
        // Merging kernels with different signatures will break invariants. E.g.
        // https://github.com/taichi-dev/taichi/blob/a6575fb97557267e2f550591f43b183076b72ac2/taichi/transforms/type_check.cpp#L326
        //
        // TODO: we could merge different kernels if their args are the same.
        // But we have no way to check that for now.
        auto check = [](const Kernel *k) {
          return (k->args.empty() && k->rets.empty());
        };
        kernel_args_match = (check(rec_i.kernel) && check(rec_j.kernel));
      }
      // TODO: avoid snode accessors going into async engine
      const bool is_snode_accessor =
          (rec_i.kernel->is_accessor || rec_j.kernel->is_accessor);
      bool fusable =
          (is_same_range_for || is_same_struct_for || are_both_serial) &&
          kernel_args_match && !is_snode_accessor;
      task_type_fusable[i][j] = fusable;
    }
  }

  auto do_fuse = [this](SFGNode *a, SFGNode *b) {
    auto &rec_a = a->rec;
    auto &rec_b = b->rec;
    // We are about to change both |task_a| and |task_b|. Clone them first.
    auto cloned_task_a = rec_a.ir_handle.clone();
    auto cloned_task_b = rec_b.ir_handle.clone();
    auto task_a = cloned_task_a->as<OffloadedStmt>();
    auto task_b = cloned_task_b->as<OffloadedStmt>();
    // TODO: in certain cases this optimization can be wrong!
    // Fuse task b into task_a
    for (int j = 0; j < (int)task_b->body->size(); j++) {
      task_a->body->insert(std::move(task_b->body->statements[j]));
    }
    task_b->body->statements.clear();

    // replace all reference to the offloaded statement B to A
    irpass::replace_all_usages_with(task_a, task_b, task_a);

    auto kernel = rec_a.kernel;
    irpass::full_simplify(task_a, /*after_lower_access=*/false, kernel);
    // For now, re_id is necessary for the hash to be correct.
    irpass::re_id(task_a);

    auto h = ir_bank_->get_hash(task_a);
    rec_a.ir_handle = IRHandle(task_a, h);
    ir_bank_->insert(std::move(cloned_task_a), h);
    rec_b.ir_handle = IRHandle(nullptr, 0);

    // TODO: since cloned_task_b->body is empty, can we remove this (i.e.,
    //  simply delete cloned_task_b here)?
    ir_bank_->insert_to_trash_bin(std::move(cloned_task_b));
  };

  auto fused = std::make_unique<bool[]>(n);

  bool modified = false;
  while (true) {
    bool updated = false;
    for (int i = 1; i < n; i++)
      fused[i] = !nodes_[i]->rec.empty();
    for (int i = 1; i < n; i++) {
      if (!fused[i]) {
        for (auto &edges : nodes_[i]->output_edges) {
          for (auto &edge : edges.second) {
            const int j = edge->node_id;
            // TODO: for each pair of edge (i, j), we can only fuse if they are
            //  serial or both element-wise.
            if (!fused[i] && !fused[j] && task_type_fusable[i][j]) {
              do_fuse(nodes_[i].get(), nodes_[j].get());
              fused[i] = fused[j] = true;
              updated = true;
            }
          }
        }
      }
    }
    if (updated) {
      modified = true;
    } else {
      break;
    }
  }

  return modified;
}

std::vector<TaskLaunchRecord> StateFlowGraph::extract() {
  std::vector<TaskLaunchRecord> tasks;
  tasks.reserve(nodes_.size());
  for (int i = 1; i < (int)nodes_.size(); i++) {
    tasks.push_back(nodes_[i]->rec);
  }
  clear();
  return tasks;
}

void StateFlowGraph::print() {
  fmt::print("=== State Flow Graph ===\n");
  for (auto &node : nodes_) {
    fmt::print("{}\n", node->string());
    if (!node->input_edges.empty()) {
      fmt::print("  Inputs:\n");
      for (const auto &p : node->input_edges) {
        for (const auto *to : p.second) {
          fmt::print("    {} <- {}\n", p.first.name(), to->string());
        }
      }
    }
    if (!node->output_edges.empty()) {
      fmt::print("  Outputs:\n");
      for (const auto &p : node->output_edges) {
        for (const auto *to : p.second) {
          fmt::print("    {} -> {}\n", p.first.name(), to->string());
        }
      }
    }
  }
  fmt::print("=======================\n");
}

std::string StateFlowGraph::dump_dot(
    const std::optional<std::string> &rankdir) {
  using SFGNode = StateFlowGraph::Node;
  using TaskType = OffloadedStmt::TaskType;
  std::stringstream ss;
  ss << "digraph {\n";
  auto node_id = [](const SFGNode *n) {
    // https://graphviz.org/doc/info/lang.html ID naming
    return fmt::format("n_{}_{}", n->task_name, n->launch_id);
  };
  // Graph level configuration.
  if (rankdir) {
    ss << "  rankdir=" << *rankdir << "\n";
  }
  ss << "\n";
  // Specify the node styles
  std::unordered_set<const SFGNode *> latest_state_nodes;
  for (const auto &p : latest_state_owner_) {
    latest_state_nodes.insert(p.second);
  }
  std::vector<const SFGNode *> nodes_with_no_inputs;
  for (const auto &nd : nodes_) {
    const auto *n = nd.get();
    ss << "  " << fmt::format("{} [label=\"{}\"", node_id(n), n->string());
    if (nd->is_initial_node) {
      ss << ",shape=box";
    } else if (latest_state_nodes.find(n) != latest_state_nodes.end()) {
      ss << ",peripheries=2";
    }
    // Highlight user-defined tasks
    const auto tt = nd->task_type;
    if (!nd->is_initial_node &&
        (tt == TaskType::range_for || tt == TaskType::struct_for ||
         tt == TaskType::serial)) {
      ss << ",style=filled,fillcolor=lightgray";
    }
    ss << "]\n";
    if (nd->input_edges.empty())
      nodes_with_no_inputs.push_back(n);
  }
  ss << "\n";
  {
    // DFS
    std::unordered_set<const SFGNode *> visited;
    std::vector<const SFGNode *> stack(nodes_with_no_inputs);
    while (!stack.empty()) {
      auto *from = stack.back();
      stack.pop_back();
      if (visited.find(from) == visited.end()) {
        visited.insert(from);
        for (const auto &p : from->output_edges) {
          for (const auto *to : p.second) {
            stack.push_back(to);
            std::string style;

            if (!from->has_state_flow(p.first, to)) {
              style = "style=dotted";
            }

            ss << "  "
               << fmt::format("{} -> {} [label=\"{}\" {}]", node_id(from),
                              node_id(to), p.first.name(), style)
               << '\n';
          }
        }
      }
    }
  }
  ss << "}\n";  // closes "dirgraph {"
  return ss.str();
}

TLANG_NAMESPACE_END
