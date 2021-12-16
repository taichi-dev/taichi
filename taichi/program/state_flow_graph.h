#pragma once

#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef TI_WITH_LLVM
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#else
#include <vector>
#include <set>
#endif
#include "taichi/ir/ir.h"
#include "taichi/lang_util.h"
#include "taichi/program/async_utils.h"
#include "taichi/program/program.h"
#include "taichi/util/bit.h"

TLANG_NAMESPACE_BEGIN

class IRBank;
class AsyncEngine;

class StateFlowGraph {
 public:
  struct Node;

  // A specialized container for fast edge insertion and lookup
  class StateToNodesMap {
   public:
    static constexpr unsigned kNumInlined = 8u;
    using Edge = std::pair<AsyncState, Node *>;
#ifdef TI_WITH_LLVM
    using Container = llvm::SmallVector<Edge, kNumInlined>;
#else
    using Container = std::vector<Edge>;
#endif

    StateToNodesMap() = default;

    bool empty() const {
      return data_.empty();
    }
    std::size_t size() const {
      return data_.size();
    }
    void clear();
    // Returns if there is an edge from the owning node to |n| through |as|.
    bool has_edge(const Edge &e) const;
    bool has_edge(const AsyncState &as, Node *n) const {
      return has_edge(std::make_pair(as, n));
    }
    bool has_state(const AsyncState &as) const;

    void insert_edge(const AsyncState &as, Node *n);
    // Removes all occurrences of Node |n|
    void remove_node(const Node *n);

    // The two methods below are dedicated for fusion.
    // If edge(as, old_nd) exists, then replaces |old_nd| with |new_nd| and
    // returns true. Otherwise returns false.
    bool replace_node_in_edge(const AsyncState &as, Node *old_nd, Node *new_nd);
    // Insert an edge when |data_| is already sorted.
    void insert_edge_sorted(const AsyncState &as, Node *n);

    struct Range {
      using iterator = StateToNodesMap::Container::iterator;

      iterator begin() {
        return begin_;
      }

      iterator end() {
        return end_;
      }

      std::size_t size() const {
        return end_ - begin_;
      }

     private:
      friend class StateToNodesMap;
      explicit Range(iterator begin, iterator end) : begin_(begin), end_(end) {
      }

      iterator begin_;
      iterator end_;
    };

    // Iterates over the edges on state |as| in sorted order.
    Range operator[](const AsyncState &as);
    // Iterates over all the edges in sorted order.
    Range get_all_edges();

    // Only iterates through the async states. This is dedicated for fusion.
    // Non-STL conforming
    class StateIterator {
     public:
      bool done() const {
        return cur_ == end_;
      }

      AsyncState get_state() const {
        TI_ASSERT(!done());
        return cur_->first;
      }

      StateIterator &operator++();  // pre increment
      // Is there an edge to node |n| through the current async state?
      bool has_edge(Node *n) const;

     private:
      friend class StateToNodesMap;
      using const_iterator = StateToNodesMap::Container::const_iterator;

      explicit StateIterator(const StateToNodesMap::Container &data)
          : cur_(data.begin()), end_(data.end()) {
      }

      const_iterator cur_;
      const_iterator end_;
    };

    StateIterator get_state_iterator() const {
      TI_ASSERT(sorted_);
      return StateIterator(data_);
    }

    // After this, one cannot insert edges anymore (unless using
    // insert_edge_sorted()).
    // |allow_already_sorted| is a backdoor for the initial node.
    void sort_edges(bool allow_already_sorted = false);

    // This reverts the container to its unsorted state (including |mask_|).
    // It's unfortunate to have this method. But without this, we cannot support
    // executed but retained nodes.
    void unsort_edges();

    // For debugging purpose
    int node_id = -1;

   private:
    bool matches(const Container::iterator &it, const Edge &e) const {
      return (it != data_.end()) && (*it == e);
    }

    static Edge get_high_sentinel(const AsyncState &as) {
      return std::make_pair(
          as, reinterpret_cast<Node *>(std::numeric_limits<uintptr_t>().max()));
    }

    static int get_mask_slot(const Edge &e);

    using MaskType = uint64;
    Container data_;
    bool sorted_{false};
    MaskType mask_{0};
  };
  // Each node is a task
  // Note: after SFG is done, each node here should hold a TaskLaunchRecord.
  // Optimization should happen fully on the SFG, instead of the queue in
  // AsyncEngine.
  // Before we migrate, the SFG is only used for visualization. Therefore we
  // only store a kernel_name, which is used as the label of the GraphViz node.
  struct Node {
    TaskLaunchRecord rec;
    TaskMeta *meta{nullptr};
    // Incremental ID to identify the i-th launch of the task.
    bool is_initial_node{false};

    // Returns the position in nodes_. Invoke StateFlowGraph::reid_nodes() to
    // keep it up-to-date.
    int node_id{0};

    // Returns the position in get_pending_tasks() or extract_pending_tasks().
    // For executed tasks (including the initial node), pending_node_id is -1.
    int pending_node_id{0};

    // Performance hits
    // * std::unordered_multimap: Slow on clang + libc++, see #1855
    // * std::unordered_map<., std::unordered_set<.>>: slow due to frequent
    //   allocation, see #1927.
    StateToNodesMap input_edges, output_edges;

    bool pending() const {
      return pending_node_id >= 0;
    }

    bool executed() const {
      return pending_node_id == -1;
    }

    void mark_executed() {
      pending_node_id = -1;
    }

    std::string string() const;

    // Note: there are two types of edges A->B:
    //   ------
    //   Dependency edge: A must execute before B
    //
    //   Flow edge: A generates some data that is consumed by B. This also
    //   implies that A must execute before B
    //
    //   Therefore Flow edge = Dependency edge + possible state flow

    bool has_state_flow(AsyncState state, const Node *destination) const {
      // True: (Flow edge) the state generated by this node is used by the
      // destination node.

      // False: (Dependency edge) the destination node does not
      // use the generated state, but its execution must happen after this case.
      // This usually means a write-after-read (WAR) dependency on state.

      // Note:
      // Read-after-write leads to flow edges
      // Write-after-write leads to dependency edges
      // Write-after-read leads to dependency edges
      //
      // So an edge is a data flow edge iff the destination node reads the
      // state.
      //

      return destination->meta->input_states.find(state) !=
             destination->meta->input_states.end();
    }

    void disconnect_all();

    void disconnect_with(Node *other);
  };

  StateFlowGraph(AsyncEngine *engine,
                 IRBank *ir_bank,
                 const CompileConfig *const config);

  std::vector<Node *> get_pending_tasks() const;

  // Returns get_pending_tasks()[begin, end).
  std::vector<Node *> get_pending_tasks(int begin, int end) const;

  std::vector<std::unique_ptr<Node>> extract_pending_tasks();

  void clear();

  void mark_pending_tasks_as_executed();

  void print();

  // Returns a string representing a DOT graph.
  //
  // |embed_states_threshold|: We can choose to embed the states into the task
  // node itself, if there aren't too many output states. This defines the
  // maximum number of output states a task can have for the states to be
  // embedded in the node.
  //
  // TODO: In case we add more and more DOT configs, create a struct?
  std::string dump_dot(const std::optional<std::string> &rankdir,
                       int embed_states_threshold = 0,
                       bool include_hash = false);

  void insert_tasks(const std::vector<TaskLaunchRecord> &rec,
                    bool filter_listgen);

  void insert_node(std::unique_ptr<Node> &&node);

  void insert_edge(Node *from, Node *to, AsyncState state);

  // Compute transitive closure for tasks in get_pending_tasks()[begin, end).
  std::pair<std::vector<bit::Bitset>, std::vector<bit::Bitset>>
  compute_transitive_closure(int begin, int end);

  // Fuse tasks in get_pending_tasks()[begin, end),
  // return the indices to delete.
  std::unordered_set<int> fuse_range(int begin, int end);

  bool fuse();

  bool optimize_listgen();

  bool demote_activation();

  bool optimize_dead_store();

  void delete_nodes(const std::unordered_set<int> &indices_to_delete);

  void reid_nodes();

  void reid_pending_nodes();

  void replace_reference(Node *node_a,
                         Node *node_b,
                         bool only_output_edges = false);

  void topo_sort_nodes();

  void sort_node_edges();

  void verify(bool also_verify_ir = false) const;

  // Extract all pending tasks and insert them in topological/original order.
  void rebuild_graph(bool sort);

  // Extract all tasks to execute.
  std::vector<TaskLaunchRecord> extract_to_execute();

  std::size_t size() const {
    return nodes_.size();
  }

  int num_pending_tasks() const {
    return nodes_.size() - first_pending_task_index_;
  }

  // Recursively mark as dirty the list state of "snode" and all its children
  void mark_list_as_dirty(SNode *snode);

  void benchmark_rebuild_graph();

  AsyncState get_async_state(SNode *snode, AsyncState::Type type);

  AsyncState get_async_state(Kernel *kernel);

  void populate_latest_state_owner(std::size_t id);
#ifdef TI_WITH_LLVM
  using LatestStateReaders =
      llvm::SmallVector<std::pair<AsyncState, llvm::SmallSet<Node *, 8>>, 4>;
#else
  using LatestStateReaders =
      std::vector<std::pair<AsyncState, std::set<Node *>>>;
#endif

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  Node *initial_node_;  // The initial node holds all the initial states.
  int first_pending_task_index_;
  TaskMeta initial_meta_;
  std::vector<Node *> latest_state_owner_;
  LatestStateReaders latest_state_readers_;
  std::unordered_map<std::string, int> task_name_to_launch_ids_;
  IRBank *ir_bank_;
  std::unordered_map<SNode *, bool> list_up_to_date_;
  [[maybe_unused]] AsyncEngine *engine_;
  const CompileConfig *const config_;
};

TLANG_NAMESPACE_END
