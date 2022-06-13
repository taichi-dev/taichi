#include "taichi/program/state_flow_graph.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_set>

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/async_engine.h"
#include "taichi/util/statistics.h"
#include "taichi/system/timeline.h"

// Keep this include in the end!
#include "taichi/program/async_profiler_switch.h"

TLANG_NAMESPACE_BEGIN

namespace {

using LatestStateReaders = StateFlowGraph::LatestStateReaders;

std::pair<LatestStateReaders::value_type::second_type *, bool> insert(
    LatestStateReaders &m,
    const AsyncState &s) {
  int sz = m.size();
  for (int i = 0; i < sz; i++) {
    if (m[i].first == s) {
      return std::make_pair(&m[i].second, true);
    }
  }
  m.push_back(std::make_pair(s, LatestStateReaders::value_type::second_type{}));
  return std::make_pair(&(m.back().second), false);
}

LatestStateReaders::value_type::second_type &get_or_insert(
    LatestStateReaders &m,
    const AsyncState &s) {
  // get_or_insert() implies that the user doesn't care whether |s| is already
  // in |m|, so we just return the mapped value. This is functionally equivalent
  // to a (unordered) map's operator[].
  return *(insert(m, s).first);
}

bool insert(LatestStateReaders &m,
            const AsyncState &s,
            StateFlowGraph::Node *n) {
  auto p = insert(m, s);
  const bool b = p.first->insert(n).second;
  return p.second && b;
}

using StateToNodesMap = StateFlowGraph::StateToNodesMap;

}  // namespace
// TODO: rename state to edge since we have not only state flow edges but also
// dependency edges.

void StateToNodesMap::clear() {
  sorted_ = false;
  mask_ = 0;
  // It seems that once we are beyond the small buffer threshold, there's no
  // going back...
  data_.clear();
}

bool StateToNodesMap::has_edge(const Edge &e) const {
  TI_ASSERT(sorted_);
  return std::binary_search(data_.begin(), data_.end(), e);
}

bool StateToNodesMap::has_state(const AsyncState &as) const {
  for (auto iter = get_state_iterator(); !iter.done(); ++iter) {
    if (iter.get_state() == as) {
      return true;
    }
  }
  return false;
}

void StateToNodesMap::insert_edge(const AsyncState &as, Node *n) {
  TI_ASSERT(!sorted_);
  const Edge e = std::make_pair(as, n);
  const auto slot = get_mask_slot(e);
  const MaskType mask_bit = (MaskType(1) << slot);
  if ((mask_ & mask_bit) == 0) {
    mask_ |= mask_bit;
  } else if (std::find(data_.begin(), data_.end(), e) != data_.end()) {
    return;
  }
  data_.push_back(e);
}

void StateToNodesMap::remove_node(const Node *n) {
  TI_ASSERT(sorted_);
  // This should be used only after |data_| is sorted.
  // 1) disconnect_with
  // 2) fuse_range
  // So adding the mask shouldn't affect this implementation.

  // TODO: Instead of linear scan, binary search by state?
  int cur = 0;
  for (int i = 0; i < data_.size(); ++i) {
    if (data_[i].second != n) {
      data_[cur++] = data_[i];
    }
  }
  data_.resize(cur);
}

bool StateToNodesMap::replace_node_in_edge(const AsyncState &as,
                                           Node *old_nd,
                                           Node *new_nd) {
  TI_ASSERT(sorted_);
  const Edge oe = std::make_pair(as, old_nd);
  auto iter = std::lower_bound(data_.begin(), data_.end(), oe);
  if (!matches(iter, oe)) {
    return false;
  }
  if (has_edge(as, new_nd)) {
    data_.erase(iter);
    return true;
  }
  iter->second = new_nd;

  // TODO: Is getting the range an overkill? Just sort the entire data range?
  auto rn = (*this)[as];
  TI_ASSERT(rn.begin() <= iter && iter < rn.end());
  std::sort(rn.begin(), rn.end());
  return true;
}

void StateToNodesMap::insert_edge_sorted(const AsyncState &as, Node *n) {
  TI_ASSERT(sorted_);

  const Edge e = std::make_pair(as, n);
  auto lb = std::lower_bound(data_.begin(), data_.end(), e);
  if (matches(lb, e)) {
    return;
  }
  data_.insert(lb, e);
}

StateToNodesMap::Range StateToNodesMap::operator[](const AsyncState &as) {
  TI_ASSERT(sorted_);

  const Edge lo_sentinel = std::make_pair(as, reinterpret_cast<Node *>(0));
  auto begin = std::lower_bound(data_.begin(), data_.end(), lo_sentinel);
  const Edge hi_sentinel = get_high_sentinel(as);
  auto end = std::upper_bound(begin, data_.end(), hi_sentinel);
  return Range(begin, end);
}

StateToNodesMap::Range StateToNodesMap::get_all_edges() {
  if (empty()) {
    // Note that we intentionally do NOT check |sorted_| if this is empty. This
    // is because fusion may call clear(), which sets |sorted_| to false. Later
    // on, the cleared node will be disconnected, at which point this method
    // gets invoked.
    //
    // More things to note:
    // * Following the above case, |sorted_| is also false for remove_node()
    // during disconnection. However, because the range is empty, remove_node()
    // is never really invoked...
    // * Unlike here, operator[] unconditionally checks for |sorted_|.
    return Range(data_.end(), data_.end());
  }
  TI_ASSERT(sorted_);

  return Range(data_.begin(), data_.end());
}

void StateToNodesMap::sort_edges(bool allow_already_sorted) {
  if (sorted_ && allow_already_sorted) {
    return;
  }
  TI_ASSERT(!sorted_);

  // TODO(mask): need to compact the inlined region first.
  std::sort(data_.begin(), data_.end());
  sorted_ = true;
}

void StateToNodesMap::unsort_edges() {
  const auto old_size = data_.size();
  auto tmp_data = std::move(data_);
  clear();
  for (const auto &e : tmp_data) {
    insert_edge(e.first, e.second);
  }
  TI_ASSERT(data_.size() == old_size);
}

StateToNodesMap::StateIterator &StateToNodesMap::StateIterator::operator++() {
  if (!done()) {
    const auto cur_state = get_state();
    // Advances to the next state
    cur_ = std::upper_bound(cur_, end_, get_high_sentinel(cur_state));
  }
  return *this;
}

bool StateToNodesMap::StateIterator::has_edge(Node *n) const {
  if (done())
    return false;
  const Edge e = std::make_pair(get_state(), n);
  // Assuming finding the upper bound for the current state matters little...
  return std::binary_search(cur_, end_, e);
}

// static
int StateToNodesMap::get_mask_slot(const Edge &e) {
  constexpr int kNumBits = sizeof(MaskType) * 8;
  // Divided by 8 to avoid trailing zeros.
  // TODO: see if e.second->node_id suits better for hashing.
  const size_t h =
      e.first.unique_id ^ (reinterpret_cast<uintptr_t>(e.second) / 8);
  return h % kNumBits;
}

std::string StateFlowGraph::Node::string() const {
  return fmt::format("[node: {}:{}]", meta->name, rec.id);
}

void StateFlowGraph::Node::disconnect_all() {
  for (auto &other : output_edges.get_all_edges()) {
    other.second->disconnect_with(this);
  }
  for (auto &other : input_edges.get_all_edges()) {
    other.second->disconnect_with(this);
  }
}

void StateFlowGraph::Node::disconnect_with(StateFlowGraph::Node *other) {
  output_edges.remove_node(other);
  input_edges.remove_node(other);
}

StateFlowGraph::StateFlowGraph(AsyncEngine *engine,
                               IRBank *ir_bank,
                               const CompileConfig *const config)
    : first_pending_task_index_(1 /*after initial node*/),
      ir_bank_(ir_bank),
      engine_(engine),
      config_(config) {
  nodes_.push_back(std::make_unique<Node>());
  initial_node_ = nodes_.back().get();
  initial_meta_.name = "initial_state";
  initial_node_->meta = &initial_meta_;
  initial_node_->is_initial_node = true;
  initial_node_->node_id = 0;
  initial_node_->input_edges.node_id = 0;
  initial_node_->output_edges.node_id = 0;
  initial_node_->mark_executed();
}

std::vector<StateFlowGraph::Node *> StateFlowGraph::get_pending_tasks() const {
  return get_pending_tasks(/*begin=*/0, /*end=*/num_pending_tasks());
}

std::vector<StateFlowGraph::Node *> StateFlowGraph::get_pending_tasks(
    int begin,
    int end) const {
  std::vector<Node *> pending_tasks;
  TI_ASSERT(0 <= begin && begin <= end);
  TI_ASSERT(end <= num_pending_tasks());
  pending_tasks.reserve(end - begin);
  for (int i = first_pending_task_index_ + begin;
       i < first_pending_task_index_ + end; i++) {
    pending_tasks.push_back(nodes_[i].get());
  }
  return pending_tasks;
}

std::vector<std::unique_ptr<StateFlowGraph::Node>>
StateFlowGraph::extract_pending_tasks() {
  std::vector<std::unique_ptr<Node>> pending_tasks;
  TI_ASSERT(nodes_.size() >= first_pending_task_index_);
  pending_tasks.reserve(nodes_.size() - first_pending_task_index_);
  for (int i = first_pending_task_index_; i < (int)nodes_.size(); i++) {
    pending_tasks.emplace_back(std::move(nodes_[i]));
  }
  nodes_.resize(first_pending_task_index_);
  return pending_tasks;
}

void StateFlowGraph::clear() {
  // TODO: GC here?
  nodes_.resize(1);  // Erase all nodes except the initial one
  initial_node_->output_edges.clear();
  std::fill(latest_state_owner_.begin(), latest_state_owner_.end(),
            initial_node_);
  latest_state_readers_.clear();
  first_pending_task_index_ = 1;

  // Do not clear task_name_to_launch_ids_.
}

void StateFlowGraph::mark_pending_tasks_as_executed() {
  std::vector<std::unique_ptr<Node>> new_nodes;
  std::unordered_set<Node *> state_owners;
  for (auto owner : latest_state_owner_) {
    state_owners.insert(state_owners.end(), owner);
  }
  for (auto &node : nodes_) {
    if (node->is_initial_node || state_owners.count(node.get()) > 0) {
      node->mark_executed();
      new_nodes.push_back(std::move(node));
    }
  }
  nodes_ = std::move(new_nodes);
  first_pending_task_index_ = nodes_.size();
  reid_nodes();
}

void StateFlowGraph::insert_tasks(const std::vector<TaskLaunchRecord> &records,
                                  bool filter_listgen) {
  TI_AUTO_PROF;
  std::vector<TaskLaunchRecord> filtered_records;
  if (filter_listgen && config_->async_opt_listgen) {
    /*
     * Here we find all the ClearList/ListGen pairs and try to evict obviously
     * redundant list operations.
     */
    for (int i = 0; i < (int)records.size(); i++) {
      auto r = records[i];
      auto meta = get_task_meta(ir_bank_, r);
      for (auto output_state : meta->output_states) {
        if (output_state.type == AsyncState::Type::mask) {
          // If the task modifies an SNode mask, all lists of all its children
          // will be marked as dirty.
          mark_list_as_dirty(output_state.snode());
        }
      }
      // push it back for now (maybe removed later if we find it is redundant)
      filtered_records.push_back(r);
      auto offload = r.ir_handle.ir()->as<OffloadedStmt>();
      auto snode = offload->snode;
      if (i < 1 || offload->task_type != OffloadedTaskType::listgen)
        continue;
      auto previous_r = records[i - 1];
      auto previous_offload = previous_r.ir_handle.ir()->as<OffloadedStmt>();
      if (previous_offload->task_type != OffloadedTaskType::serial ||
          previous_offload->body->statements.size() != 1) {
        TI_ERROR(
            "When early filtering tasks, the previous task of list "
            "generation must be a serial offload with a single statement.");
      }
      auto clear_list =
          previous_offload->body->statements[0]->cast<ClearListStmt>();
      if (!clear_list || clear_list->snode != snode)
        TI_ERROR("Invalid clear list stmt");
      stat.add("total_list_gen");
      if (list_up_to_date_[snode]) {
        stat.add("filtered_list_gen");
        // Remove the list gen task
        filtered_records.pop_back();
        // Remove the clear list task
        filtered_records.pop_back();
      } else {
        list_up_to_date_[snode] = true;
      }
    }
  } else {
    filtered_records = records;
  }
  for (const auto &rec : filtered_records) {
    auto node = std::make_unique<Node>();
    node->rec = rec;
    node->meta = get_task_meta(ir_bank_, rec);
    insert_node(std::move(node));
  }
}

void StateFlowGraph::insert_node(std::unique_ptr<StateFlowGraph::Node> &&node) {
  for (auto input_state : node->meta->input_states) {
    insert_edge(latest_state_owner_[input_state.unique_id], node.get(),
                input_state);
  }
  for (auto output_state : node->meta->output_states) {
    if (get_or_insert(latest_state_readers_, output_state).empty()) {
      if (latest_state_owner_[output_state.unique_id] != initial_node_) {
        // insert a WAW dependency edge
        insert_edge(latest_state_owner_[output_state.unique_id], node.get(),
                    output_state);
      } else {
        insert(latest_state_readers_, output_state, initial_node_);
      }
    }
    latest_state_owner_[output_state.unique_id] = node.get();
    for (auto *d : get_or_insert(latest_state_readers_, output_state)) {
      // insert a WAR dependency edge
      insert_edge(d, node.get(), output_state);
    }
    get_or_insert(latest_state_readers_, output_state).clear();
  }
  // Note that this loop must happen AFTER the previous one
  for (auto input_state : node->meta->input_states) {
    insert(latest_state_readers_, input_state, node.get());
  }
  nodes_.push_back(std::move(node));
}

void StateFlowGraph::insert_edge(Node *from, Node *to, AsyncState state) {
  TI_ASSERT(from != nullptr);
  TI_ASSERT(to != nullptr);
  from->output_edges.insert_edge(state, to);
  to->input_edges.insert_edge(state, from);
}

bool StateFlowGraph::optimize_listgen() {
  TI_AUTO_PROF;
  TI_AUTO_TIMELINE;
  bool modified = false;

  std::vector<std::pair<int, int>> common_pairs;

  topo_sort_nodes();

  std::unordered_map<SNode *, std::vector<Node *>> listgen_nodes;

  for (int i = 1; i < nodes_.size(); i++) {
    auto node = nodes_[i].get();
    if (node->meta->type == OffloadedStmt::TaskType::listgen)
      listgen_nodes[node->meta->snode].push_back(node);
  }

  std::unordered_set<int> nodes_to_delete;

  for (auto &record : listgen_nodes) {
    auto &listgens = record.second;

    // Thanks to the dependency edges, the order of nodes in listgens is
    // UNIQUE. (Consider the list state of the SNode.)

    // Note that there can be > 1 executed listgens because they are
    // latest state readers of other states.
    int i_start = 0;
    while (i_start + 1 < listgens.size() && listgens[i_start + 1]->executed()) {
      i_start++;
    }

    // We can only replace a continuous subset of listgen entries.
    // So the nested loop below is actually O(n).
    for (int i = i_start; i < listgens.size(); i++) {
      auto node_a = listgens[i];

      [[maybe_unused]] bool erased_any = false;

      auto new_i = i;

      for (int j = i + 1; j < listgens.size(); j++) {
        auto node_b = listgens[j];
        TI_ASSERT(!node_b->executed());

        // Test if two list generations share the same mask and parent list
        auto snode = node_a->meta->snode;

        auto list_state = get_async_state(snode, AsyncState::Type::list);
        auto parent_list_state =
            get_async_state(snode->parent, AsyncState::Type::list);

        if (snode->need_activation()) {
          // Needs mask state
          auto mask_state = get_async_state(snode, AsyncState::Type::mask);
          TI_ASSERT(node_a->input_edges[mask_state].size() == 1);
          TI_ASSERT(node_b->input_edges[mask_state].size() == 1);

          if (node_a->input_edges[mask_state].begin()->second !=
              node_b->input_edges[mask_state].begin()->second)
            break;
        }

        TI_ASSERT(node_a->input_edges[parent_list_state].size() == 1);
        TI_ASSERT(node_b->input_edges[parent_list_state].size() == 1);
        if (node_a->input_edges[parent_list_state].begin()->second !=
            node_b->input_edges[parent_list_state].begin()->second)
          break;

        TI_ASSERT(node_b->input_edges[list_state].size() == 1);
        Node *clear_node = node_b->input_edges[list_state].begin()->second;
        TI_ASSERT(!clear_node->executed());
        // TODO: This could be a bottleneck, avoid unnecessary IR clone.
        // However, the task most likely will only contain a single
        // ClearListStmt, so it's not a big deal...
        auto new_ir = clear_node->rec.ir_handle.clone();
        auto *new_clear_list_task = new_ir->as<OffloadedStmt>();
        TI_ASSERT(new_clear_list_task->task_type ==
                  OffloadedStmt::TaskType::serial);
        auto &stmts = new_clear_list_task->body->statements;
        auto pred = [snode](const pStmt &s) -> bool {
          auto *cls = s->cast<ClearListStmt>();
          return (cls != nullptr && cls->snode == snode);
        };
        // There should only be one clear list for |node_b|.
        TI_ASSERT(std::count_if(stmts.begin(), stmts.end(), pred) == 1);
        auto cls_it = std::find_if(stmts.begin(), stmts.end(), pred);
        stmts.erase(cls_it);

        if (stmts.empty()) {
          // Just erase the empty serial task that used to hold ClearListStmt
          nodes_to_delete.insert(clear_node->node_id);
        } else {
          // IR modified. Node should be updated.
          auto new_handle =
              IRHandle(new_ir.get(), ir_bank_->get_hash(new_ir.get()));
          ir_bank_->insert(std::move(new_ir), new_handle.hash());
          clear_node->rec.ir_handle = new_handle;
          clear_node->meta = get_task_meta(ir_bank_, clear_node->rec);
        }

        TI_DEBUG("Common list generation {} and (to erase) {}",
                 node_a->string(), node_b->string());

        nodes_to_delete.insert(node_b->node_id);
        erased_any = true;
        new_i = j;
      }
      i = new_i;
    }
  }

  if (!nodes_to_delete.empty()) {
    modified = true;
    delete_nodes(nodes_to_delete);
    // Note: DO NOT topo sort the nodes here. Node deletion destroys order
    // independency.
    rebuild_graph(/*sort=*/false);
  }

  return modified;
}

std::pair<std::vector<bit::Bitset>, std::vector<bit::Bitset>>
StateFlowGraph::compute_transitive_closure(int begin, int end) {
  TI_AUTO_PROF;
  using bit::Bitset;
  const int n = end - begin;
  auto nodes = get_pending_tasks(begin, end);

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
    for (auto &edge : nodes[i]->input_edges.get_all_edges()) {
      auto tmp_id = edge.second->pending_node_id - begin;
      if (tmp_id >= 0 && tmp_id < n) {
        TI_ASSERT(tmp_id < i);
        has_path[tmp_id] |= has_path[i];
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto &edge : nodes[i]->output_edges.get_all_edges()) {
      auto tmp_id = edge.second->pending_node_id - begin;
      if (tmp_id >= 0 && tmp_id < n) {
        TI_ASSERT(tmp_id > i);
        has_path_reverse[tmp_id] |= has_path_reverse[i];
      }
    }
  }
  return std::make_pair(std::move(has_path), std::move(has_path_reverse));
}

std::unordered_set<int> StateFlowGraph::fuse_range(int begin, int end) {
  TI_AUTO_PROF;
  using bit::Bitset;
  const int n = end - begin;
  if (n <= 1) {
    return std::unordered_set<int>();
  }

  auto nodes = get_pending_tasks(begin, end);

  std::vector<Bitset> has_path, has_path_reverse;
  std::tie(has_path, has_path_reverse) = compute_transitive_closure(begin, end);

  // Classify tasks by TaskFusionMeta.
  std::vector<TaskFusionMeta> fusion_meta(n);
  // It seems that std::set is slightly faster than std::unordered_set here.
  std::unordered_map<TaskFusionMeta, std::set<int>> task_fusion_map;
  for (int i = 0; i < n; i++) {
    fusion_meta[i] = get_task_fusion_meta(ir_bank_, nodes[i]->rec);
    if (fusion_meta[i].fusible) {
      auto &fusion_set = task_fusion_map[fusion_meta[i]];
      fusion_set.insert(fusion_set.end(), i);
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
        has_path[j][i] = true;
      }
    }
  };

  auto fused = std::vector<bool>(n);

  auto do_fuse = [&](int a, int b) {
    TI_PROFILER("do_fuse");
    TI_ASSERT(0 <= a && a < b && b < n);
    auto *node_a = nodes[a];
    auto *node_b = nodes[b];
    TI_TRACE("Fuse: nodes[{}]({}) <- nodes[{}]({})", a, node_a->string(), b,
             node_b->string());
    stat.add("num_fused_tasks");
    auto &rec_a = node_a->rec;
    auto &rec_b = node_b->rec;
    rec_a.ir_handle =
        ir_bank_->fuse(rec_a.ir_handle, rec_b.ir_handle, rec_a.kernel);
    rec_b.ir_handle = IRHandle();

    // Convert to the index in nodes_.
    indices_to_delete.insert(b + begin + first_pending_task_index_);

    const bool already_had_a_to_b_edge = has_path[a][b];
    if (already_had_a_to_b_edge) {
      node_a->output_edges.remove_node(node_b);
      node_b->input_edges.remove_node(node_a);
    }
    replace_reference(node_b, node_a);

    // update the transitive closure
    insert_edge_for_transitive_closure(b, a);
    if (!already_had_a_to_b_edge)
      insert_edge_for_transitive_closure(a, b);

    fused[b] = true;
    task_fusion_map[fusion_meta[a]].erase(a);
    task_fusion_map[fusion_meta[b]].erase(b);
  };

  auto edge_fusible = [&](int a, int b) {
    TI_PROFILER("edge_fusible");
    TI_ASSERT(a >= 0 && a < nodes.size() && b >= 0 && b < nodes.size());
    // Check if a and b are fusible if there is an edge (a, b).
    if (fused[a] || fused[b] || !fusion_meta[a].fusible ||
        fusion_meta[a] != fusion_meta[b]) {
      return false;
    }
    if (nodes[a]->meta->type != OffloadedTaskType::serial) {
      std::unordered_map<int, int> offload_map;
      offload_map[0] = 0;
      std::unordered_set<AsyncState> modified_states =
          get_task_meta(ir_bank_, nodes[a]->rec)->output_states;
      std::unordered_set<AsyncState> modified_states_b =
          get_task_meta(ir_bank_, nodes[a]->rec)->output_states;
      modified_states.insert(modified_states_b.begin(),
                             modified_states_b.end());
      AsyncStateSet modified_states_set{modified_states};
      for (auto state_iter = nodes[a]->output_edges.get_state_iterator();
           !state_iter.done(); ++state_iter) {
        auto state = state_iter.get_state();
        if (!state.holds_snode()) {
          continue;
        }
        auto *snode = state.snode();
        const auto sty = state.type;
        if (sty != AsyncState::Type::value && sty != AsyncState::Type::mask) {
          // No need to check allocator/list states, as they must be accompanied
          // with either value or mask states.
          continue;
        }
        if (state_iter.has_edge(nodes[b])) {
          if (nodes[a]->meta->loop_unique.count(snode) == 0 ||
              nodes[b]->meta->loop_unique.count(snode) == 0) {
            return false;
          }
          auto same_loop_unique_address = [&](GlobalPtrStmt *ptr1,
                                              GlobalPtrStmt *ptr2) {
            if (!ptr1 || !ptr2) {
              return false;
            }
            TI_ASSERT(snode == ptr1->snodes[0]);
            TI_ASSERT(snode == ptr2->snodes[0]);
            TI_ASSERT(nodes[a]->rec.stmt()->id == 0);
            TI_ASSERT(nodes[b]->rec.stmt()->id == 0);
            // Only map the OffloadedStmt to see if both SNodes are loop-unique
            // on the same statement.
            for (int i = 0; i < (int)ptr1->indices.size(); i++) {
              if (!irpass::analysis::same_value(
                      ptr1->indices[i], ptr2->indices[i], modified_states_set,
                      ir_bank_,
                      std::make_optional<std::unordered_map<int, int>>(
                          offload_map))) {
                return false;
              }
            }
            return true;
          };
          if (!same_loop_unique_address(nodes[a]->meta->loop_unique[snode],
                                        nodes[b]->meta->loop_unique[snode])) {
            return false;
          }
        }
      }
    }
    // check if a doesn't have a path to b of length >= 2
    auto a_has_path_to_b = has_path[a] & has_path_reverse[b];
    a_has_path_to_b[a] = a_has_path_to_b[b] = false;
    return a_has_path_to_b.none();
  };

  for (int i = 0; i < n; i++) {
    fused[i] = nodes[i]->rec.empty();
  }
  const auto initial_remaining_fuses_per_task =
      (config_->async_max_fuse_per_task > 0) ? config_->async_max_fuse_per_task
                                             : n;

  // The case without an edge: O(sum(size * min(size, n / 64))) = O(n^2 / 64)
  const int kLargeFusionSetThreshold = std::max(n / 16, 16);
  for (auto &fusion_map : task_fusion_map) {
    std::vector<int> indices(fusion_map.second.begin(),
                             fusion_map.second.end());
    TI_ASSERT(std::is_sorted(indices.begin(), indices.end()));
    if (indices.size() >= kLargeFusionSetThreshold) {
      // O(size * n / 64)
      Bitset mask(n);
      for (int a : indices) {
        mask[a] = !fused[a];
      }
      for (int a : indices) {
        if (!fused[a]) {
          // Fuse no more than remaining_fuses tasks into task a,
          // otherwise do_fuse may be very slow
          Bitset current_mask = (mask & ~(has_path[a] | has_path_reverse[a]));
          int b = a + 1;
          int remaining_fuses = initial_remaining_fuses_per_task;
          while ((b != -1) && (remaining_fuses > 0)) {
            b = current_mask.lower_bound(b);
            if (b == -1) {
              mask[a] = false;  // a can't be fused in this iteration
            } else {
              do_fuse(a, b);
              mask[a] = false;
              mask[b] = false;
              b++;  // for computing the next lower bound
              remaining_fuses--;
            }
          }
        }
      }
    } else if (indices.size() >= 2) {
      // O(size^2)
      std::vector<int> start_index(indices.size());
      for (int i = 0; i < (int)indices.size(); i++) {
        start_index[i] = i + 1;
      }
      while (true) {
        bool done = true;
        for (int i = 0; i < (int)indices.size(); i++) {
          const int a = indices[i];
          if (!fused[a]) {
            int remaining_fuses = initial_remaining_fuses_per_task;
            // Fuse no more than remaining_fuses tasks into task a
            // in this iteration
            for (int &j = start_index[i];
                 (j < (int)indices.size()) && (remaining_fuses > 0); j++) {
              const int b = indices[j];
              if (!fused[b] && !has_path[a][b] && !has_path[b][a]) {
                do_fuse(a, b);
                remaining_fuses--;
              }
            }
            if (start_index[i] < (int)indices.size()) {
              done = false;
            }
          }
        }
        if (done) {
          break;
        }
      }
    }
  }
  // The case with an edge: O(nm / 64)
  for (int i = 0; i < n; i++) {
    if (!fused[i]) {
      // Fuse no more than one task into task i
      for (auto &edge : nodes[i]->output_edges.get_all_edges()) {
        if (edge.second->pending()) {
          const int j = edge.second->pending_node_id - begin;
          if (j >= 0 && j < nodes.size() && edge_fusible(i, j)) {
            do_fuse(i, j);
            // Iterators of nodes[i]->output_edges may be invalidated
            break;
          }
        }
      }
    }
  }

  return indices_to_delete;
}

bool StateFlowGraph::fuse() {
  TI_AUTO_PROF;
  TI_AUTO_TIMELINE;
  using bit::Bitset;
  // Only guarantee to fuse tasks with indices in nodes_ differ by less than
  // kMaxFusionDistance if there are too many tasks.
  const int kMaxFusionDistance = 512;

  // Invoke fuse_range() <= floor(num_pending_tasks() / kMaxFusionDistance)
  // times with (end - begin) <= 2 * kMaxFusionDistance.
  std::unordered_set<int> indices_to_delete;
  const int n = num_pending_tasks();
  if (true) {
    indices_to_delete = fuse_range(0, n);
  } else {
    // TODO: fuse by range
    for (int i = 0; i < n; i += kMaxFusionDistance * 2) {
      auto indices = fuse_range(i, std::min(n, i + kMaxFusionDistance * 2));
      indices_to_delete.insert(indices.begin(), indices.end());
    }
    if (indices_to_delete.empty()) {
      for (int i = kMaxFusionDistance; i < n; i += kMaxFusionDistance * 2) {
        auto indices = fuse_range(i, std::min(n, i + kMaxFusionDistance * 2));
        indices_to_delete.insert(indices.begin(), indices.end());
      }
    }
  }

  bool modified = !indices_to_delete.empty();
  // TODO: Do we need a trash bin here?
  if (modified) {
    // Rebuild the graph in topological order.
    // The original order may not be a correct topological order.
    delete_nodes(indices_to_delete);
    rebuild_graph(/*sort=*/true);
  }

  return modified;
}

void StateFlowGraph::rebuild_graph(bool sort) {
  TI_AUTO_PROF;
  TI_AUTO_TIMELINE;
  if (sort)
    topo_sort_nodes();
  std::vector<TaskLaunchRecord> tasks;
  tasks.reserve(nodes_.size());
  int num_executed_tasks = 0;
  for (int i = 1; i < (int)nodes_.size(); i++) {
    if (!nodes_[i]->rec.empty()) {
      tasks.push_back(nodes_[i]->rec);
      if (nodes_[i]->executed())
        num_executed_tasks++;
    }
  }
  clear();
  insert_tasks(tasks, false);
  for (int i = 1; i <= num_executed_tasks; i++) {
    nodes_[i]->mark_executed();
  }
  first_pending_task_index_ = num_executed_tasks + 1;
  reid_nodes();
  reid_pending_nodes();

  sort_node_edges();
}

std::vector<TaskLaunchRecord> StateFlowGraph::extract_to_execute() {
  TI_AUTO_PROF;
  auto nodes = get_pending_tasks();
  std::vector<TaskLaunchRecord> tasks;
  tasks.reserve(nodes.size());
  for (auto &node : nodes) {
    if (!node->rec.empty()) {
      tasks.push_back(node->rec);
    }
  }
  mark_pending_tasks_as_executed();
  rebuild_graph(/*sort=*/false);
  for (int i = 0; i < first_pending_task_index_; ++i) {
    // The reason we do this is that, upon the next launch, we could insert
    // edges to these executed but retained nodes. To allow for insertion, we
    // have to "unsort" them here.
    nodes_[i]->input_edges.unsort_edges();
    nodes_[i]->output_edges.unsort_edges();
  }
  return tasks;
}

void StateFlowGraph::print() {
  fmt::print("=== State Flow Graph ===\n");
  fmt::print("{} nodes ({} pending)\n", size(), num_pending_tasks());
  for (auto &node : nodes_) {
    fmt::print("{}{}\n", node->string(), node->executed() ? " (executed)" : "");
    if (!node->input_edges.empty()) {
      fmt::print("  Inputs:\n");
      for (const auto &to : node->input_edges.get_all_edges()) {
        fmt::print("    {} <- {}\n", to.first.name(), to.second->string());
      }
    }
    if (!node->output_edges.empty()) {
      fmt::print("  Outputs:\n");
      for (const auto &to : node->output_edges.get_all_edges()) {
        fmt::print("    {} -> {}\n", to.first.name(), to.second->string());
      }
    }
  }
  fmt::print("=======================\n");
}

std::string StateFlowGraph::dump_dot(const std::optional<std::string> &rankdir,
                                     int embed_states_threshold,
                                     bool include_hash) {
  using SFGNode = StateFlowGraph::Node;
  using TaskType = OffloadedStmt::TaskType;
  std::stringstream ss;

  // TODO: expose an API that allows users to highlight a single state
  AsyncState highlight_state =
      get_async_state(nullptr, AsyncState::Type::value);

  ss << "digraph {\n";
  const std::string fontname = "consolas";
  ss << "  graph [fontname = \"" << fontname << "\"]\n";
  ss << "  node [fontname = \"" << fontname << "\"]\n";
  ss << "  edge [fontname = \"" << fontname << "\"]\n";
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
  for (auto p : latest_state_owner_) {
    latest_state_nodes.insert(p);
  }

  bool highlight_single_state = false;

  auto node_selected = [&](const SFGNode *node) {
    if (highlight_single_state) {
      return node->input_edges.has_state(highlight_state) ||
             node->output_edges.has_state(highlight_state);
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
  for (auto &nd : nodes_) {
    const auto *n = nd.get();

    std::stringstream labels;
    // No states embedded.
    labels << escaped_label(n->string());
    if (include_hash && !n->is_initial_node) {
      labels << fmt::format("\\nhash: 0x{:08x}", n->rec.ir_handle.hash());
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
           tt == TaskType::mesh_for || tt == TaskType::serial)) {
        // ss << " style=filled fillcolor=lightgray";
      }
      ss << "]\n";
    }
    if (nd->input_edges.empty())
      nodes_with_no_inputs.push_back(n);
  }
  ss << "\n";
  for (const auto &n : nodes_) {
    auto *from = n.get();
    for (const auto &edge : from->output_edges.get_all_edges()) {
      const auto *to = edge.second;
      const bool states_embedded = (nodes_with_embedded_states.find(from) !=
                                    nodes_with_embedded_states.end());
      std::string from_node_port = node_id(from);
      std::stringstream attribs;
      if (states_embedded) {
        // The state is embedded inside the node. We draw the edge from
        // the port corresponding to this state.
        // Format is "{node}:{port}"
        from_node_port += fmt::format(":{}", edge.first.name());
      } else {
        // Show the state on the edge label
        attribs << fmt::format("label=\"{}\"", edge.first.name());
      }

      if (!from->has_state_flow(edge.first, to)) {
        attribs << " style=dotted";
      }

      if (highlight_single_state) {
        if (state_selected(edge.first)) {
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
  ss << "}\n";  // closes "dirgraph {"
  return ss.str();
}

void StateFlowGraph::topo_sort_nodes() {
  TI_AUTO_PROF
  // Only sort pending tasks.
  const auto previous_size = nodes_.size();
  std::deque<std::unique_ptr<Node>> queue;
  std::vector<int> degrees_in(num_pending_tasks());

  reid_pending_nodes();

  auto pending_tasks = extract_pending_tasks();
  for (auto &node : pending_tasks) {
    int degree_in = 0;
    for (auto &input_edge : node->input_edges.get_all_edges()) {
      if (input_edge.second->pending()) {
        degree_in++;
      }
    }
    degrees_in[node->pending_node_id] = degree_in;
  }

  for (auto &node : pending_tasks) {
    if (degrees_in[node->pending_node_id] == 0) {
      queue.emplace_back(std::move(node));
    }
  }

  while (!queue.empty()) {
    auto head = std::move(queue.front());
    queue.pop_front();

    // Delete the node and update degrees_in
    for (auto &output_edge : head->output_edges.get_all_edges()) {
      auto dest = output_edge.second->pending_node_id;
      TI_ASSERT(dest != -1);
      degrees_in[dest]--;
      TI_ASSERT_INFO(degrees_in[dest] >= 0, "dest={} degrees_in={}", dest,
                     degrees_in[dest]);
      if (degrees_in[dest] == 0) {
        queue.push_back(std::move(pending_tasks[dest]));
      }
    }

    nodes_.emplace_back(std::move(head));
  }

  if (previous_size != nodes_.size()) {
    auto first_not_sorted = std::find_if_not(
        degrees_in.begin(), degrees_in.end(), [](int x) { return x == 0; });
    if (first_not_sorted != degrees_in.end()) {
      TI_ERROR("SFG contains a cycle with node {}{}",
               first_not_sorted - degrees_in.begin(),
               pending_tasks[first_not_sorted - degrees_in.begin()]->string());
    } else {
      TI_ERROR(
          "Topo sort changes the size of SFG ({} -> {}) "
          "but no cycles are found",
          previous_size, nodes_.size());
    }
  }
  reid_nodes();
  reid_pending_nodes();
}

void StateFlowGraph::sort_node_edges() {
  // Note that inside clear(), |initial_node_| is preserved, including its
  // input/output edges. Therefore it could be already sorted across different
  // SFG stages. For other nodes, each stage destroys them all and reallocate
  // new ones.
  initial_node_->input_edges.sort_edges(/*allow_already_sorted=*/true);
  initial_node_->output_edges.sort_edges(/*allow_already_sorted=*/true);
  for (int i = 1; i < nodes_.size(); ++i) {
    auto &nd = nodes_[i];
    nd->input_edges.sort_edges();
    nd->output_edges.sort_edges();
  }
}

void StateFlowGraph::reid_nodes() {
  TI_AUTO_PROF
  for (int i = 0; i < nodes_.size(); i++) {
    nodes_[i]->node_id = i;
    nodes_[i]->input_edges.node_id = i;
    nodes_[i]->output_edges.node_id = i;
  }
  TI_ASSERT(initial_node_->node_id == 0);
}

void StateFlowGraph::reid_pending_nodes() {
  TI_AUTO_PROF
  for (int i = first_pending_task_index_; i < nodes_.size(); i++) {
    nodes_[i]->pending_node_id = i - first_pending_task_index_;
  }
}

void StateFlowGraph::replace_reference(StateFlowGraph::Node *node_a,
                                       StateFlowGraph::Node *node_b,
                                       bool only_output_edges) {
  TI_AUTO_PROF
  // replace all edges to node A with new ones to node B
  for (auto &edge : node_a->output_edges.get_all_edges()) {
    // Find all nodes C that points to A
    // Replace reference to A with B
    const auto &ostate = edge.first;
    auto *node_c = edge.second;
    TI_ASSERT_INFO(node_c != node_b,
                   "Edge {} --({})-> {} will become a self-loop "
                   "after replacing reference",
                   node_a->string(), ostate.name(), node_b->string());
    if (node_c->input_edges.replace_node_in_edge(ostate, /*old_nd=*/node_a,
                                                 /*new_nd=*/node_b)) {
      // Use the sorted version of edge insertion!
      node_b->output_edges.insert_edge_sorted(ostate, node_c);
    }
  }
  node_a->output_edges.clear();
  if (only_output_edges) {
    return;
  }
  for (auto &edge : node_a->input_edges.get_all_edges()) {
    // Find all nodes C that points to A
    // Replace reference to A with B
    const auto &istate = edge.first;
    auto *node_c = edge.second;
    TI_ASSERT_INFO(node_c != node_b,
                   "Edge {} <-({})-- {} will become a self-loop "
                   "after replacing reference",
                   node_a->string(), istate.name(), node_b->string());
    if (node_c->output_edges.replace_node_in_edge(istate, /*old_nd=*/node_a,
                                                  /*new_nd=*/node_b)) {
      // Use the sorted version of edge insertion!
      node_b->input_edges.insert_edge_sorted(istate, node_c);
    }
  }
  node_a->input_edges.clear();
}

void StateFlowGraph::delete_nodes(
    const std::unordered_set<int> &indices_to_delete) {
  TI_AUTO_PROF
  std::vector<std::unique_ptr<Node>> new_nodes_;
  std::unordered_set<Node *> nodes_to_delete;

  for (auto &i : indices_to_delete) {
    TI_ASSERT(nodes_[i]->pending());
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
    if (nodes_to_delete.find(s) != nodes_to_delete.end()) {
      s = initial_node_;
    }
  }

  for (auto &s : latest_state_readers_) {
    for (auto n : nodes_to_delete) {
      s.second.erase(n);
    }
  }

  nodes_ = std::move(new_nodes_);
  reid_nodes();
  reid_pending_nodes();
}

bool StateFlowGraph::optimize_dead_store() {
  TI_AUTO_PROF
  TI_AUTO_TIMELINE;
  bool modified = false;

  auto nodes = get_pending_tasks();
  for (auto &task : nodes) {
    // Dive into this task and erase dead stores
    std::set<const SNode *> store_eliminable_snodes;
    // Try to find unnecessary output state
    for (auto &s : task->meta->output_states) {
      if (s.type != AsyncState::Type::value || !s.holds_snode()) {
        // Listgen elimination has been handled in optimize_listgen, so we will
        // only focus on "value" states.
        continue;
      }
      if (latest_state_owner_[s.unique_id] == task) {
        // Cannot eliminate the latest write, because it may form a state-flow
        // with the later kernel launches.
        //
        // TODO: Add some sort of hints so that the compiler knows that some
        // value will never be used?
        continue;
      }
      auto *snode = s.snode();
      bool used = false;
      for (auto &other : task->output_edges[s]) {
        if (task->has_state_flow(s, other.second)) {
          // Check if this is a RAW dependency. For scalar SNodes, a
          // A WAW flow edge decades to a dependency edge.
          used = true;
        } else {
          // Note that a dependency edge does not count as an data usage
          if (!snode->is_scalar() &&
              other.second->meta->element_wise.count(snode) == 0) {
            // If the SNode is not element-wise written, the value stored
            // in this task can be used later.
            used = true;
          }
        }
      }
      // This state is used by some other node, so it cannot be erased
      if (used) {
        continue;
      }

      store_eliminable_snodes.insert(snode);
    }

    // *****************************
    // Erase the state s output.
    if (!store_eliminable_snodes.empty()) {
      const bool verbose = task->rec.kernel->program->config.verbose;

      const auto dse_result = ir_bank_->optimize_dse(
          task->rec.ir_handle, store_eliminable_snodes, verbose);
      auto new_handle = dse_result.first;
      if (new_handle != task->rec.ir_handle) {
        modified = true;
        task->rec.ir_handle = new_handle;
        task->meta = get_task_meta(ir_bank_, task->rec);
      }
      bool first_compute = !dse_result.second;
      if (first_compute && modified) {
        stat.add("sfg_dse_tasks", 1.0);
      }
      if (first_compute && verbose) {
        // Log only for the first time, otherwise we will be overwhelmed very
        // quickly...
        std::vector<std::string> snodes_strs;
        for (const auto *sn : store_eliminable_snodes) {
          snodes_strs.push_back(sn->get_node_type_name_hinted());
        }
        TI_INFO("SFG DSE: task={} snodes={} optimized?={}", task->string(),
                fmt::join(snodes_strs, ", "), modified);
      }
    }
  }

  std::unordered_set<int> to_delete;
  // erase empty blocks
  for (int i = 0; i < (int)nodes.size(); i++) {
    auto &meta = *nodes[i]->meta;
    auto ir = nodes[i]->rec.ir_handle.ir()->cast<OffloadedStmt>();
    const auto mt = meta.type;
    // Do NOT check ir->body->statements first! |ir->body| could be done when
    // |mt| is not the desired type.
    if ((mt == OffloadedTaskType::serial ||
         mt == OffloadedTaskType::struct_for ||
         mt == OffloadedTaskType::mesh_for ||
         mt == OffloadedTaskType::range_for) &&
        ir->body->statements.empty()) {
      to_delete.insert(i + first_pending_task_index_);
    }
  }

  // Task metas can change even if the number of tasks doesn't change.
  if (modified) {
    delete_nodes(to_delete);
    rebuild_graph(/*sort=*/false);
  }

  return modified;
}

void StateFlowGraph::verify(bool also_verify_ir) const {
#ifndef ASYNC_ENGINE_RELEASE_MODE
  TI_AUTO_PROF
  // Check nodes
  const int n = nodes_.size();
  TI_ASSERT_INFO(n >= 1, "SFG is empty");
  for (int i = 0; i < n; i++) {
    TI_ASSERT_INFO(nodes_[i], "nodes_[{}] is nullptr", i);
  }
  TI_ASSERT_INFO(nodes_[0]->is_initial_node,
                 "nodes_[0] is not the initial node");
  TI_ASSERT_INFO(nodes_[0].get() == initial_node_,
                 "initial_node_ is not nodes_[0]");
  TI_ASSERT(first_pending_task_index_ <= n);
  for (int i = 0; i < first_pending_task_index_; i++) {
    TI_ASSERT_INFO(nodes_[i]->pending_node_id == -1,
                   "nodes_[{}]({})->pending_node_id is {} (should be -1)", i,
                   nodes_[i]->string(), nodes_[i]->pending_node_id);
  }
  for (int i = first_pending_task_index_; i < n; i++) {
    TI_ASSERT_INFO(nodes_[i]->pending_node_id == i - first_pending_task_index_,
                   "nodes_[{}]({})->pending_node_id is {} (should be {})", i,
                   nodes_[i]->string(), nodes_[i]->pending_node_id,
                   i - first_pending_task_index_);
  }
  for (int i = 0; i < n; i++) {
    TI_ASSERT_INFO(nodes_[i]->node_id == i, "nodes_[{}]({})->node_id is {}", i,
                   nodes_[i]->string(), nodes_[i]->node_id);
  }

  // Check edges
  for (int i = 0; i < n; i++) {
    for (auto &edge : nodes_[i]->output_edges.get_all_edges()) {
      auto *node = edge.second;
      TI_ASSERT_INFO(node, "nodes_[{}]({}) has an empty output edge", i,
                     nodes_[i]->string());
      auto dest = node->node_id;
      TI_ASSERT_INFO(dest >= 0 && dest < n,
                     "nodes_[{}]({}) has an output edge to nodes_[{}]", i,
                     nodes_[i]->string(), dest);
      TI_ASSERT_INFO(nodes_[dest].get() == node,
                     "nodes_[{}]({}) has an output edge to {}, "
                     "which is outside nodes_",
                     i, nodes_[i]->string(), node->string());
      TI_ASSERT_INFO(dest != i, "nodes_[{}]({}) has an output edge to itself",
                     i, nodes_[i]->string());
      TI_ASSERT_INFO(
          nodes_[dest]->input_edges.has_edge(edge.first, nodes_[i].get()),
          "nodes_[{}]({}) has an output edge to nodes_[{}]({}), "
          "which doesn't corresponds to an input edge",
          i, nodes_[i]->string(), dest, nodes_[dest]->string());
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto &edge : nodes_[i]->input_edges.get_all_edges()) {
      auto *node = edge.second;
      TI_ASSERT_INFO(node, "nodes_[{}]({}) has an empty input edge", i,
                     nodes_[i]->string());
      auto dest = node->node_id;
      TI_ASSERT_INFO(dest >= 0 && dest < n,
                     "nodes_[{}]({}) has an input edge to nodes_[{}]", i,
                     nodes_[i]->string(), dest);
      TI_ASSERT_INFO(nodes_[dest].get() == node,
                     "nodes_[{}]({}) has an input edge to {}, "
                     "which is outside nodes_",
                     i, nodes_[i]->string(), node->string());
      TI_ASSERT_INFO(dest != i, "nodes_[{}]({}) has an input edge to itself", i,
                     nodes_[i]->string());
      TI_ASSERT_INFO(
          nodes_[dest]->output_edges.has_edge(edge.first, nodes_[i].get()),
          "nodes_[{}]({}) has an input edge to nodes_[{}]({}), "
          "which doesn't corresponds to an output edge",
          i, nodes_[i]->string(), dest, nodes_[dest]->string());
    }
  }

  // Check topological order
  for (int i = 0; i < n; i++) {
    for (auto &edge : nodes_[i]->output_edges.get_all_edges()) {
      auto dest = edge.second->node_id;
      TI_ASSERT_INFO(dest > i,
                     "topological order violated: nodes_[{}]({}) "
                     "has an output edge to nodes_[{}]({})",
                     i, nodes_[i]->string(), dest, nodes_[dest]->string());
    }
  }

  if (also_verify_ir) {
    // Check IR
    for (int i = 1; i < n; i++) {
      TI_ASSERT_INFO(
          nodes_[i]->meta->type == nodes_[i]->rec.stmt()->task_type,
          "nodes_[{}]({}) has type {}, "
          "but its IR has task_type {}",
          i, nodes_[i]->string(),
          offloaded_task_type_name(nodes_[i]->meta->type),
          offloaded_task_type_name(nodes_[i]->rec.stmt()->task_type));
      irpass::analysis::verify(nodes_[i]->rec.stmt());
    }
  }
#endif  // ASYNC_ENGINE_RELEASE_MODE
}

bool StateFlowGraph::demote_activation() {
  TI_AUTO_PROF
  TI_AUTO_TIMELINE;
  bool modified = false;

  topo_sort_nodes();

  // TODO: use unordered_map
  std::map<std::pair<IRHandle, Node *>, std::vector<Node *>> tasks;

  // Gather identical struct-for tasks that use the same lists
  for (int i = 1; i < (int)nodes_.size(); i++) {
    Node *node = nodes_[i].get();
    auto snode = node->meta->snode;
    auto list_state = get_async_state(snode, AsyncState::Type::list);

    // Currently we handle struct for only
    // TODO: handle serial and range for
    if (node->meta->type != OffloadedTaskType::struct_for)
      continue;

    if (node->input_edges[list_state].size() != 1)
      continue;

    auto *list_node = node->input_edges[list_state].begin()->second;
    tasks[std::make_pair(node->rec.ir_handle, list_node)].push_back(node);
  }

  for (auto &task : tasks) {
    auto &nodes = task.second;
    TI_ASSERT(nodes.size() > 0);
    if (nodes.size() <= 1)
      continue;

    // Starting from the second task in the list, activations may be demoted
    auto new_handle = ir_bank_->demote_activation(nodes[0]->rec.ir_handle);
    if (new_handle != nodes[0]->rec.ir_handle) {
      modified = true;
      TI_ASSERT(!nodes[1]->executed());
      nodes[1]->rec.ir_handle = new_handle;
      nodes[1]->meta = get_task_meta(ir_bank_, nodes[1]->rec);
      // Copy nodes[1] replacement result to later nodes
      for (int j = 2; j < (int)nodes.size(); j++) {
        TI_ASSERT(!nodes[j]->executed());
        nodes[j]->rec.ir_handle = new_handle;
        nodes[j]->meta = nodes[1]->meta;
      }
      // For every "demote_activation" call, we only optimize for a single key
      // in std::map<std::pair<IRHandle, Node *>, std::vector<Node *>> tasks
      // since the graph probably needs to be rebuild after demoting
      // part of the tasks.
      break;
    }
  }

  if (modified) {
    rebuild_graph(/*sort=*/false);
  }

  return modified;
}

void StateFlowGraph::mark_list_as_dirty(SNode *snode) {
  list_up_to_date_[snode] = false;
  for (auto &ch : snode->ch) {
    if (ch->type != SNodeType::place) {
      mark_list_as_dirty(ch.get());
    }
  }
}

void StateFlowGraph::benchmark_rebuild_graph() {
  double total_time = 0;
  for (int k = 0; k < 100000; k++) {
    auto t = Time::get_time();
    for (int i = 0; i < 100; i++)
      rebuild_graph(/*sort=*/false);
    auto rebuild_t = Time::get_time() - t;
    total_time += rebuild_t;
    TI_INFO(
        "nodes = {} total time {:.4f} ms (averaged so far {:.4} ms); per_node "
        "{:.4f} ns",
        nodes_.size(), rebuild_t * 1e4, (total_time * 1e4 / (k + 1)),
        1e7 * rebuild_t / nodes_.size());
  }
}

AsyncState StateFlowGraph::get_async_state(SNode *snode,
                                           AsyncState::Type type) {
  return ir_bank_->get_async_state(snode, type);
}

AsyncState StateFlowGraph::get_async_state(Kernel *kernel) {
  return ir_bank_->get_async_state(kernel);
}

void StateFlowGraph::populate_latest_state_owner(std::size_t id) {
  if (id >= latest_state_owner_.size()) {
    std::size_t old_size = latest_state_owner_.size();
    latest_state_owner_.resize(id + 1);
    for (int i = old_size; i < latest_state_owner_.size(); i++) {
      latest_state_owner_[i] = initial_node_;
    }
  }
}

TLANG_NAMESPACE_END
