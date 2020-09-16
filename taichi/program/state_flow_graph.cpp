#include "taichi/program/state_flow_graph.h"

#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/program/async_engine.h"
#include "state_flow_graph.h"

#include <sstream>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// TODO: rename state to edge since we have not only state flow edges but also
// dependency edges.

std::string StateFlowGraph::Node::string() const {
  return fmt::format("[node: {}:{}]", meta->name, rec.id);
}

void StateFlowGraph::Node::disconnect_all() {
  for (auto &edges : output_edges) {
    for (auto &other : edges.second) {
      other->disconnect_with(this);
    }
  }
  for (auto &edges : input_edges) {
    for (auto &other : edges.second) {
      other->disconnect_with(this);
    }
  }
}

void StateFlowGraph::Node::disconnect_with(StateFlowGraph::Node *other) {
  for (auto &edges : output_edges) {
    edges.second.erase(other);
  }
  for (auto &edges : input_edges) {
    edges.second.erase(other);
  }
}

StateFlowGraph::StateFlowGraph(IRBank *ir_bank) : ir_bank_(ir_bank) {
  nodes_.push_back(std::make_unique<Node>());
  initial_node_ = nodes_.back().get();
  initial_meta_.name = "initial_state";
  initial_node_->meta = &initial_meta_;
  initial_node_->is_initial_node = true;
}

void StateFlowGraph::clear() {
  // TODO: GC here?
  nodes_.resize(1);  // Erase all nodes except the initial one
  initial_node_->output_edges.clear();
  latest_state_owner_.clear();
  latest_state_readers_.clear();

  // Do not clear task_name_to_launch_ids_.
}

void StateFlowGraph::insert_task(const TaskLaunchRecord &rec) {
  auto node = std::make_unique<Node>();
  node->rec = rec;
  node->meta = get_task_meta(ir_bank_, rec);
  for (auto input_state : node->meta->input_states) {
    if (latest_state_owner_.find(input_state) == latest_state_owner_.end()) {
      latest_state_owner_[input_state] = initial_node_;
    }
    insert_state_flow(latest_state_owner_[input_state], node.get(),
                      input_state);
  }
  for (auto output_state : node->meta->output_states) {
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
  for (auto input_state : node->meta->input_states) {
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

bool StateFlowGraph::optimize_listgen() {
  bool modified = false;

  std::vector<std::pair<int, int>> common_pairs;

  topo_sort_nodes();
  reid_nodes();

  std::unordered_map<SNode *, std::vector<Node *>> listgen_nodes;

  for (int i = 1; i < nodes_.size(); i++) {
    auto node = nodes_[i].get();
    if (node->meta->type == OffloadedStmt::TaskType::listgen)
      listgen_nodes[node->meta->snode].push_back(node);
  }

  std::unordered_set<int> nodes_to_delete;

  for (auto &record : listgen_nodes) {
    auto &listgens = record.second;

    // Thanks to the dependency edges, the order of nodes in listgens seems to
    // be UNIQUE
    // TODO: prove

    // We can only replace a bunch continuous entries of listgens
    for (int i = 0; i < listgens.size(); i++) {
      auto node_a = listgens[i];

      bool erasing = false;

      for (int j = i + 1; j < listgens.size(); j++) {
        auto node_b = listgens[j];

        // Test if two list generations share the same mask and parent list
        auto snode = node_a->meta->snode;

        auto mask_state = AsyncState{snode, AsyncState::Type::mask};
        auto list_state = AsyncState{snode, AsyncState::Type::list};
        auto parent_list_state =
            AsyncState{snode->parent, AsyncState::Type::list};

        TI_ASSERT(node_a->input_edges[mask_state].size() == 1);
        TI_ASSERT(node_b->input_edges[mask_state].size() == 1);

        if (*node_a->input_edges[mask_state].begin() !=
            *node_b->input_edges[mask_state].begin())
          break;

        TI_ASSERT(node_a->input_edges[parent_list_state].size() == 1);
        TI_ASSERT(node_b->input_edges[parent_list_state].size() == 1);
        if (*node_a->input_edges[parent_list_state].begin() !=
            *node_b->input_edges[parent_list_state].begin())
          break;

        TI_ASSERT(node_b->input_edges[list_state].size() == 1);
        Node *clear_node = *node_b->input_edges[list_state].begin();

        const OffloadedStmt *clear_node_offload =
            clear_node->rec.ir_handle.ir()->as<OffloadedStmt>();
        TI_ASSERT(clear_node_offload->body->statements.size() == 1);
        TI_ASSERT(clear_node_offload->body->statements[0]->is<ClearListStmt>());

        // erase the serial task containing ClearListStmt
        nodes_to_delete.insert(clear_node->node_id);

        TI_DEBUG("Common list generation {} and (to erase) {}",
                 node_a->string(), node_b->string());

        nodes_to_delete.insert(node_b->node_id);
        erasing = true;
      }

      if (erasing)
        break;
    }
  }

  TI_ASSERT(nodes_to_delete.size() % 2 == 0);

  if (!nodes_to_delete.empty()) {
    modified = true;
    delete_nodes(nodes_to_delete);
    // Note: DO NOT topo sort the nodes here. Node deletion destroys order
    // independency.
    auto tasks = extract(/*sort=*/false);
    for (auto &task : tasks) {
      insert_task(task);
    }
  }

  return modified;
}

std::pair<std::vector<bit::Bitset>, std::vector<bit::Bitset>>
StateFlowGraph::compute_transitive_closure() {
  using bit::Bitset;
  const int n = nodes_.size();
  reid_nodes();
  auto has_path = std::vector<Bitset>(n);
  auto has_path_reverse = std::vector<Bitset>(n);
  // has_path[i][j] denotes if there is a path from i to j.
  // has_path_reverse[i][j] denotes if there is a path from j to i.
  for (int i = 0; i < n; i++) {
    has_path[i] = Bitset(n);
    has_path[i][i] = true;
    has_path_reverse[i] = Bitset(n);
    has_path_reverse[i][i] = true;
  }
  for (int i = n - 1; i >= 0; i--) {
    for (auto &edges : nodes_[i]->input_edges) {
      for (auto &edge : edges.second) {
        TI_ASSERT(edge->node_id < i);
        has_path[edge->node_id] |= has_path[i];
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto &edges : nodes_[i]->output_edges) {
      for (auto &edge : edges.second) {
        // Assume nodes are sorted in topological order.
        TI_ASSERT(edge->node_id > i);
        has_path_reverse[edge->node_id] |= has_path_reverse[i];
      }
    }
  }
  return std::make_pair(std::move(has_path), std::move(has_path_reverse));
}

bool StateFlowGraph::fuse() {
  using bit::Bitset;
  const int n = nodes_.size();
  if (n <= 2) {
    return false;
  }

  std::vector<Bitset> has_path, has_path_reverse;
  std::tie(has_path, has_path_reverse) = compute_transitive_closure();

  // Cache the result that if each pair is fusable by task types.
  // TODO: improve this
  auto task_type_fusable = std::make_unique<Bitset[]>(n);
  for (int i = 0; i < n; i++) {
    task_type_fusable[i] = Bitset(n);
  }
  // nodes_[0] is the initial node.
  for (int i = 1; i < n; i++) {
    auto &rec_i = nodes_[i]->rec;
    if (rec_i.empty()) {
      continue;
    }
    auto *task_i = rec_i.stmt();
    for (int j = i + 1; j < n; j++) {
      auto &rec_j = nodes_[j]->rec;
      if (rec_j.empty()) {
        continue;
      }
      auto *task_j = rec_j.stmt();
      bool is_same_struct_for =
          task_i->task_type == OffloadedStmt::struct_for &&
          task_j->task_type == OffloadedStmt::struct_for &&
          task_i->snode == task_j->snode &&
          task_i->block_dim == task_j->block_dim;
      // TODO: a few problems with the range-for test condition:
      // 1. This could incorrectly fuse two range-for kernels that have
      // different sizes, but then the loop ranges get padded to the same
      // power-of-two (E.g. maybe a side effect when a struct-for is demoted
      // to range-for).
      // 2. It has also fused range-fors that have the same linear range,
      // but are of different dimensions of loop indices, e.g. (16, ) and
      // (4, 4).
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
        // Merging kernels with different signatures will break invariants.
        // E.g.
        // https://github.com/taichi-dev/taichi/blob/a6575fb97557267e2f550591f43b183076b72ac2/taichi/transforms/type_check.cpp#L326
        //
        // TODO: we could merge different kernels if their args are the
        // same. But we have no way to check that for now.
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

  std::unordered_set<int> indices_to_delete;

  auto insert_edge_for_transitive_closure = [&](int a, int b) {
    // insert edge a -> b
    auto update_list = has_path[a].or_eq_get_update_list(has_path[b]);
    for (auto i : update_list) {
      auto update_list_i =
          has_path_reverse[i].or_eq_get_update_list(has_path_reverse[a]);
      for (auto j : update_list_i) {
        has_path[i][j] = true;
      }
    }
  };

  auto do_fuse = [&](int a, int b) {
    auto *node_a = nodes_[a].get();
    auto *node_b = nodes_[b].get();
    // TODO: remove debug output
    TI_TRACE("Fuse: {} <- {}", node_a->string(), node_b->string());
    auto &rec_a = node_a->rec;
    auto &rec_b = node_b->rec;
    rec_a.ir_handle =
        ir_bank_->fuse(rec_a.ir_handle, rec_b.ir_handle, rec_a.kernel);
    rec_b.ir_handle = IRHandle();

    indices_to_delete.insert(b);

    const bool already_had_a_to_b_edge = has_path[a][b];
    if (already_had_a_to_b_edge) {
      for (auto &edges : node_a->output_edges) {
        edges.second.erase(node_b);
      }
      for (auto &edges : node_b->input_edges) {
        edges.second.erase(node_a);
      }
    }
    replace_reference(node_b, node_a);

    // update the transitive closure
    insert_edge_for_transitive_closure(b, a);
    if (!already_had_a_to_b_edge)
      insert_edge_for_transitive_closure(a, b);
  };

  auto fused = std::make_unique<bool[]>(n);

  bool modified = false;
  while (true) {
    bool updated = false;
    for (int i = 1; i < n; i++) {
      fused[i] = nodes_[i]->rec.empty();
    }
    for (int i = 1; i < n; i++) {
      if (!fused[i]) {
        bool i_updated = false;
        for (auto &edges : nodes_[i]->output_edges) {
          for (auto &edge : edges.second) {
            const int j = edge->node_id;
            // TODO: for each pair of edge (i, j), we can only fuse if they
            // are both serial or both element-wise.
            if (!fused[j] && task_type_fusable[i][j]) {
              auto i_has_path_to_j = has_path[i] & has_path_reverse[j];
              i_has_path_to_j[i] = i_has_path_to_j[j] = false;
              // check if i doesn't have a path to j of length >= 2
              if (i_has_path_to_j.none()) {
                do_fuse(i, j);
                fused[i] = fused[j] = true;
                i_updated = true;
                updated = true;
                break;
              }
            }
          }
          if (i_updated)
            break;
        }
      }
    }
    // TODO: accelerate this
    for (int i = 1; i < n; i++) {
      if (!fused[i]) {
        for (int j = i + 1; j < n; j++) {
          if (!fused[j] && task_type_fusable[i][j] && !has_path[i][j] &&
              !has_path[j][i]) {
            do_fuse(i, j);
            fused[i] = fused[j] = true;
            updated = true;
            break;
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

  // TODO: Do we need a trash bin here?
  if (modified) {
    // rebuild the graph in topological order
    delete_nodes(indices_to_delete);
    // TODO: we may want to preserve the original node order. Maybe
    // topo_sort_nodes() here leads to wrong results.
    topo_sort_nodes();
    auto tasks = extract();
    for (auto &task : tasks) {
      insert_task(task);
    }
  }

  return modified;
}

std::vector<TaskLaunchRecord> StateFlowGraph::extract(bool sort) {
  if (sort)
    topo_sort_nodes();
  std::vector<TaskLaunchRecord> tasks;
  tasks.reserve(nodes_.size());
  for (int i = 1; i < (int)nodes_.size(); i++) {
    if (!nodes_[i]->rec.empty()) {
      tasks.push_back(nodes_[i]->rec);

      if (false) {
        // debug
        TI_INFO("task {}:{}", nodes_[i]->meta->name, nodes_[i]->rec.id);
        nodes_[i]->meta->print();
        irpass::print(const_cast<IRNode *>(nodes_[i]->rec.ir_handle.ir()));
      }
    }
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

std::string StateFlowGraph::dump_dot(const std::optional<std::string> &rankdir,
                                     int embed_states_threshold) {
  using SFGNode = StateFlowGraph::Node;
  using TaskType = OffloadedStmt::TaskType;
  std::stringstream ss;

  // Highlight S9dense_list
  AsyncState highlight_state{get_current_program().snodes[9],
                             AsyncState::Type::list};

  ss << "digraph {\n";
  auto node_id = [](const SFGNode *n) {
    // https://graphviz.org/doc/info/lang.html ID naming
    return fmt::format("n_{}_{}", n->meta->name, n->rec.id);
  };

  auto escaped_label = [](const std::string &s) {
    std::stringstream ss;
    for (char c : s) {
      // Braces, vertical bars ,angle brackets and spaces needs to be escaped.
      // Just escape whitespaces for now...
      if (c == ' ') {
        ss << '\\';
      }
      ss << c;
    }
    return ss.str();
  };
  // Graph level configuration.
  if (rankdir) {
    ss << "  rankdir=" << *rankdir << "\n";
  }
  ss << "\n";
  // Specify the node styles
  std::unordered_set<const SFGNode *> latest_state_nodes;
  std::unordered_set<const SFGNode *> nodes_with_embedded_states;
  // TODO: make this configurable
  for (const auto &p : latest_state_owner_) {
    latest_state_nodes.insert(p.second);
  }

  bool highlight_single_state = false;

  auto node_selected = [&](const SFGNode *node) {
    if (highlight_single_state) {
      return node->input_edges.find(highlight_state) !=
                 node->input_edges.end() ||
             node->output_edges.find(highlight_state) !=
                 node->output_edges.end();
    } else {
      return true;
    }
  };

  auto state_selected = [&](AsyncState state) {
    if (highlight_single_state) {
      return state == highlight_state;
    } else {
      return true;
    }
  };

  std::vector<const SFGNode *> nodes_with_no_inputs;
  for (const auto &nd : nodes_) {
    const auto *n = nd.get();

    std::stringstream labels;
    if (!n->is_initial_node && !n->output_edges.empty() &&
        (n->output_edges.size() < embed_states_threshold)) {
      // Example:
      //
      // |-----------------------|
      // |        node foo       |
      // |-----------------------|
      // |   X_mask  |  X_value  |
      // |-----------------------|
      //
      // label={ node\ foo | { <X_mask> X_mask | <X_value> X_value } }
      // See DOT node port...
      labels << "{ " << escaped_label(n->string()) << " | { ";
      const auto &edges = n->output_edges;
      for (auto it = edges.begin(); it != edges.end(); ++it) {
        if (it != edges.begin()) {
          labels << " | ";
        }
        const auto name = it->first.name();
        // Each state corresponds to one port
        // "<port> displayed\ text"
        labels << "<" << name << "> " << escaped_label(name);
      }
      labels << " } }";

      nodes_with_embedded_states.insert(n);
    } else {
      // No states embedded.
      labels << escaped_label(n->string());
      if (!n->is_initial_node) {
        labels << fmt::format("\\nhash: 0x{:08x}", n->rec.ir_handle.hash());
      }
    }

    if (node_selected(nd.get())) {
      std::string color;
      if (highlight_single_state)
        color = " style=filled fillcolor=red ";

      ss << "  "
         << fmt::format("{} [label=\"{}\" shape=record {}", node_id(n),
                        labels.str(), color);
      if (latest_state_nodes.find(n) != latest_state_nodes.end()) {
        ss << " peripheries=2";
      }
      // Highlight user-defined tasks
      const auto tt = nd->meta->type;
      if (!nd->is_initial_node &&
          (tt == TaskType::range_for || tt == TaskType::struct_for ||
           tt == TaskType::serial)) {
        // ss << " style=filled fillcolor=lightgray";
      }
      ss << "]\n";
    }
    if (nd->input_edges.empty())
      nodes_with_no_inputs.push_back(n);
  }
  ss << "\n";
  {
    // DFS to draw edges
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

            const bool states_embedded =
                (nodes_with_embedded_states.find(from) !=
                 nodes_with_embedded_states.end());
            std::string from_node_port = node_id(from);
            std::stringstream attribs;
            if (states_embedded) {
              // The state is embedded inside the node. We draw the edge from
              // the port corresponding to this state.
              // Format is "{node}:{port}"
              from_node_port += fmt::format(":{}", p.first.name());
            } else {
              // Show the state on the edge label
              attribs << fmt::format("label=\"{}\"", p.first.name());
            }

            if (!from->has_state_flow(p.first, to)) {
              attribs << " style=dotted";
            }

            if (highlight_single_state) {
              if (state_selected(p.first)) {
                attribs << " penwidth=5 color=red";
              } else {
                attribs << " color=lightgrey";
              }
            }

            if (node_selected(from) && node_selected(to)) {
              ss << "  "
                 << fmt::format("{} -> {} [{}]", from_node_port, node_id(to),
                                attribs.str())
                 << '\n';
            }
          }
        }
      }
    }
    TI_WARN_IF(
        visited.size() > nodes_.size(),
        "Visited more nodes than what we actually have. The graph may be "
        "malformed.");
  }
  ss << "}\n";  // closes "dirgraph {"
  return ss.str();
}

void StateFlowGraph::topo_sort_nodes() {
  std::deque<std::unique_ptr<Node>> queue;
  std::vector<std::unique_ptr<Node>> new_nodes;
  std::vector<int> degrees_in(nodes_.size());

  reid_nodes();

  for (auto &node : nodes_) {
    int degree_in = 0;
    for (auto &inputs : node->input_edges) {
      degree_in += (int)inputs.second.size();
    }
    degrees_in[node->node_id] = degree_in;
  }

  for (auto &node : nodes_) {
    if (degrees_in[node->node_id] == 0) {
      queue.emplace_back(std::move(node));
    }
  }

  while (!queue.empty()) {
    auto head = std::move(queue.front());
    queue.pop_front();

    // Delete the node and update degrees_in
    for (auto &output_edge : head->output_edges) {
      for (auto &e : output_edge.second) {
        auto dest = e->node_id;
        degrees_in[dest]--;
        TI_ASSERT(degrees_in[dest] >= 0);
        if (degrees_in[dest] == 0) {
          queue.push_back(std::move(nodes_[dest]));
        }
      }
    }

    new_nodes.emplace_back(std::move(head));
  }

  TI_ASSERT(new_nodes.size() == nodes_.size());
  nodes_ = std::move(new_nodes);
  reid_nodes();
}

void StateFlowGraph::reid_nodes() {
  for (int i = 0; i < nodes_.size(); i++) {
    nodes_[i]->node_id = i;
  }
  TI_ASSERT(initial_node_->node_id == 0);
}

void StateFlowGraph::replace_reference(StateFlowGraph::Node *node_a,
                                       StateFlowGraph::Node *node_b,
                                       bool only_output_edges) {
  // replace all edges to node A with new ones to node B
  for (auto &edges : node_a->output_edges) {
    // Find all nodes C that points to A
    for (auto &node_c : edges.second) {
      // Replace reference to A with B
      if (node_c->input_edges[edges.first].find(node_a) !=
          node_c->input_edges[edges.first].end()) {
        node_c->input_edges[edges.first].erase(node_a);

        node_c->input_edges[edges.first].insert(node_b);
        node_b->output_edges[edges.first].insert(node_c);
      }
    }
  }
  node_a->output_edges.clear();
  if (only_output_edges) {
    return;
  }
  for (auto &edges : node_a->input_edges) {
    // Find all nodes C that points to A
    for (auto &node_c : edges.second) {
      // Replace reference to A with B
      if (node_c->output_edges[edges.first].find(node_a) !=
          node_c->output_edges[edges.first].end()) {
        node_c->output_edges[edges.first].erase(node_a);

        node_c->output_edges[edges.first].insert(node_b);
        node_b->input_edges[edges.first].insert(node_c);
      }
    }
  }
  node_a->input_edges.clear();
}

void StateFlowGraph::delete_nodes(
    const std::unordered_set<int> &indices_to_delete) {
  std::vector<std::unique_ptr<Node>> new_nodes_;
  std::unordered_set<Node *> nodes_to_delete;

  for (auto &i : indices_to_delete) {
    nodes_[i]->disconnect_all();
    nodes_to_delete.insert(nodes_[i].get());
  }

  for (int i = 0; i < (int)nodes_.size(); i++) {
    if (indices_to_delete.find(i) == indices_to_delete.end()) {
      new_nodes_.push_back(std::move(nodes_[i]));
    } else {
      TI_DEBUG("Deleting node {}", i);
    }
  }

  for (auto &s : latest_state_owner_) {
    if (nodes_to_delete.find(s.second) != nodes_to_delete.end()) {
      s.second = initial_node_;
    }
  }

  for (auto &s : latest_state_readers_) {
    for (auto n : nodes_to_delete) {
      s.second.erase(n);
    }
  }

  nodes_ = std::move(new_nodes_);
  reid_nodes();
}

bool StateFlowGraph::optimize_dead_store() {
  bool modified = false;

  for (int i = 1; i < nodes_.size(); i++) {
    // Start from 1 to skip the initial node

    // Dive into this task and erase dead stores
    auto &task = nodes_[i];
    // Try to find unnecessary output state
    for (auto &s : task->meta->output_states) {
      bool used = false;
      for (auto other : task->output_edges[s]) {
        if (task->has_state_flow(s, other)) {
          used = true;
        } else {
          // Note that a dependency edge does not count as an data usage
        }
      }
      // This state is used by some other node, so it cannot be erased
      if (used)
        continue;

      if (s.type != AsyncState::Type::list &&
          latest_state_owner_[s] == task.get())
        // Note that list state is special. Since a future list generation
        // always comes with ClearList, we can erase the list state even if it
        // is latest.
        continue;

      // *****************************
      // Erase the state s output.
      if (s.type == AsyncState::Type::list &&
          task->meta->type == OffloadedStmt::TaskType::serial) {
        // Try to erase list gen
        DelayedIRModifier mod;

        auto new_ir = task->rec.ir_handle.clone();
        irpass::analysis::gather_statements(new_ir.get(), [&](Stmt *stmt) {
          if (auto clear_list = stmt->cast<ClearListStmt>()) {
            if (clear_list->snode == s.snode) {
              mod.erase(clear_list);
            }
          }
          return false;
        });
        if (mod.modify_ir()) {
          // IR modified. Node should be updated.
          auto handle =
              IRHandle(new_ir.get(), ir_bank_->get_hash(new_ir.get()));
          ir_bank_->insert(std::move(new_ir), handle.hash());
          task->rec.ir_handle = handle;
          // task->meta->print();
          task->meta = get_task_meta(ir_bank_, task->rec);
          // task->meta->print();

          for (auto other : task->output_edges[s])
            other->input_edges[s].erase(task.get());

          task->output_edges.erase(s);
          modified = true;
        }
      }
    }
  }

  std::unordered_set<int> to_delete;
  // erase empty blocks
  for (int i = 1; i < (int)nodes_.size(); i++) {
    auto &meta = *nodes_[i]->meta;
    auto ir = nodes_[i]->rec.ir_handle.ir()->cast<OffloadedStmt>();
    if (meta.type == OffloadedStmt::serial && ir->body->statements.empty()) {
      to_delete.insert(i);
    } else if (meta.type == OffloadedStmt::struct_for &&
               ir->body->statements.empty()) {
      to_delete.insert(i);
    } else if (meta.type == OffloadedStmt::range_for &&
               ir->body->statements.empty()) {
      to_delete.insert(i);
    }
  }

  if (!to_delete.empty())
    modified = true;

  delete_nodes(to_delete);

  return modified;
}

void StateFlowGraph::verify() {
  const int n = nodes_.size();
  reid_nodes();
  for (int i = 0; i < n; i++) {
    for (auto &edges : nodes_[i]->output_edges) {
      for (auto &edge : edges.second) {
        TI_ASSERT_INFO(edge, "nodes_[{}]({}) has an empty output edge", i,
                       nodes_[i]->string());
        auto dest = edge->node_id;
        TI_ASSERT_INFO(dest >= 0 && dest < n,
                       "nodes_[{}]({}) has an output edge to nodes_[{}]", i,
                       nodes_[i]->string(), dest);
        TI_ASSERT_INFO(nodes_[dest].get() == edge,
                       "nodes_[{}]({}) has an output edge to {}, "
                       "which is outside nodes_",
                       i, nodes_[i]->string(), edge->string());
        TI_ASSERT_INFO(dest != i, "nodes_[{}]({}) has an output edge to itself",
                       i, nodes_[i]->string());
        auto &corresponding_edges = nodes_[dest]->input_edges[edges.first];
        TI_ASSERT_INFO(corresponding_edges.find(nodes_[i].get()) !=
                           corresponding_edges.end(),
                       "nodes_[{}]({}) has an output edge to nodes_[{}]({}), "
                       "which doesn't corresponds to an input edge",
                       i, nodes_[i]->string(), dest, nodes_[dest]->string());
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto &edges : nodes_[i]->input_edges) {
      for (auto &edge : edges.second) {
        TI_ASSERT_INFO(edge, "nodes_[{}]({}) has an empty input edge", i,
                       nodes_[i]->string());
        auto dest = edge->node_id;
        TI_ASSERT_INFO(dest >= 0 && dest < n,
                       "nodes_[{}]({}) has an input edge to nodes_[{}]", i,
                       nodes_[i]->string(), dest);
        TI_ASSERT_INFO(nodes_[dest].get() == edge,
                       "nodes_[{}]({}) has an input edge to {}, "
                       "which is outside nodes_",
                       i, nodes_[i]->string(), edge->string());
        TI_ASSERT_INFO(dest != i, "nodes_[{}]({}) has an input edge to itself",
                       i, nodes_[i]->string());
        auto &corresponding_edges = nodes_[dest]->output_edges[edges.first];
        TI_ASSERT_INFO(corresponding_edges.find(nodes_[i].get()) !=
                           corresponding_edges.end(),
                       "nodes_[{}]({}) has an input edge to nodes_[{}]({}), "
                       "which doesn't corresponds to an output edge",
                       i, nodes_[i]->string(), dest, nodes_[dest]->string());
      }
    }
  }
  // Call topological sort to check cycles.
  topo_sort_nodes();
}

// TODO: make this an IR pass
class ConstExprPropagation {
 public:
  static std::unordered_set<Stmt *> run(
      Block *block,
      const std::function<bool(Stmt *)> &is_const_seed) {
    std::unordered_set<Stmt *> const_stmts;

    auto is_const = [&](Stmt *stmt) {
      if (is_const_seed(stmt)) {
        return true;
      } else {
        return const_stmts.find(stmt) != const_stmts.end();
      }
    };

    for (auto &s : block->statements) {
      if (is_const(s.get())) {
        const_stmts.insert(s.get());
      } else if (auto binary = s->cast<BinaryOpStmt>()) {
        if (is_const(binary->lhs) && is_const(binary->rhs)) {
          const_stmts.insert(s.get());
        }
      } else if (auto unary = s->cast<UnaryOpStmt>()) {
        if (is_const(unary->operand)) {
          const_stmts.insert(s.get());
        }
      } else {
        // TODO: ...
      }
    }

    return const_stmts;
  }
};

bool StateFlowGraph::activation_demotion() {
  bool modified = false;

  topo_sort_nodes();

  std::unordered_map<IRHandle, std::vector<Node *>> tasks;

  for (int i = 1; i < (int)nodes_.size(); i++) {
    Node *node = nodes_[i].get();
    // TODO: handle serial and range for
    if (node->meta->type == OffloadedStmt::struct_for) {
      tasks[node->rec.ir_handle].push_back(node);
    }
  }

  TI_TAG;

  for (auto &task : tasks) {
    auto &nodes = task.second;
    TI_ASSERT(nodes.size() > 0);

    auto snode = nodes[0]->meta->snode;

    auto list_state = AsyncState(snode, AsyncState::Type::list);

    TI_ASSERT(snode != nullptr);

    // TODO: speed it up
    for (int i = 0; i < (int)nodes.size(); i++) {
      bool demoted = false;
      for (int j = i + 1; j < (int)nodes.size(); j++) {
        // Two nodes must use the same list state
        // TODO: for bitmasked we also need to deal with masks
        if (nodes[i]->input_edges[list_state].size() != 1)
          continue;
        if (nodes[j]->input_edges[list_state].size() != 1)
          continue;
        if (*nodes[i]->input_edges[list_state].begin() !=
            *nodes[j]->input_edges[list_state].begin())
          continue;

        auto new_ir = nodes[j]->rec.ir_handle.clone();
        OffloadedStmt *offload = new_ir->as<OffloadedStmt>();
        Block *body = offload->body.get();

        // TODO: for now we only deal with the top level. Extend.
        auto consts = ConstExprPropagation::run(body, [](Stmt *stmt) {
          if (stmt->is<ConstStmt>()) {
            return true;
          } else if (stmt->is<LoopIndexStmt>())
            return true;
          return false;
        });

        // irpass::print(offload);

        // TI_P(consts.size());

        for (int k = 0; k < (int)body->statements.size(); k++) {
          Stmt *stmt = body->statements[k].get();
          if (auto ptr = stmt->cast<GlobalPtrStmt>(); ptr && ptr->activate) {
            bool can_deactivate = true;
            for (auto ind : ptr->indices) {
              if (consts.find(ind) == consts.end()) {
                // non-constant index
                can_deactivate = false;
              }
            }
            if (can_deactivate) {
              modified = true;
              ptr->activate = false;
              demoted = true;
            }
          }
        }
        if (demoted) {
          TI_TAG;
          auto handle =
              IRHandle(new_ir.get(), ir_bank_->get_hash(new_ir.get()));
          ir_bank_->insert(std::move(new_ir), handle.hash());
          nodes[j]->rec.ir_handle = handle;
          // task->meta->print();
          nodes[j]->meta = get_task_meta(ir_bank_, nodes[j]->rec);
          break;
        }
      }
      if (demoted)
        break;
    }
  }

  if (modified) {
    auto tasks = extract(/*sort=*/false);
    for (auto &task : tasks) {
      insert_task(task);
    }
  }

  return modified;
}

void async_print_sfg() {
  get_current_program().async_engine->sfg->print();
}

std::string async_dump_dot(std::optional<std::string> rankdir,
                           int embed_states_threshold) {
  // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#allow-prohibiting-none-arguments
  return get_current_program().async_engine->sfg->dump_dot(
      rankdir, embed_states_threshold);
}

TLANG_NAMESPACE_END
