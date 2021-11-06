#include <set>
#include <unordered_map>
#include <vector>

#include "taichi/program/async_utils.h"

TLANG_NAMESPACE_BEGIN

class StateFlowGraph;

class IRBank {
 public:
  uint64 get_hash(IRNode *ir);
  void set_hash(IRNode *ir, uint64 hash);

  bool insert(std::unique_ptr<IRNode> &&ir, uint64 hash);
  void insert_to_trash_bin(std::unique_ptr<IRNode> &&ir);
  IRNode *find(IRHandle ir_handle);

  // Fuse handle_b into handle_a
  IRHandle fuse(IRHandle handle_a, IRHandle handle_b, Kernel *kernel);

  IRHandle demote_activation(IRHandle handle);

  // Try running DSE optimization on the IR identified by |handle|. |snodes|
  // denotes the set of SNodes whose stores are safe to eliminate.
  //
  // Returns:
  // * IRHandle: the (possibly) DSE-optimized IRHandle
  // * bool: whether the result is already cached.
  std::pair<IRHandle, bool> optimize_dse(IRHandle handle,
                                         const std::set<const SNode *> &snodes,
                                         bool verbose);

  std::unordered_map<IRHandle, TaskMeta> meta_bank_;
  std::unordered_map<IRHandle, TaskFusionMeta> fusion_meta_bank_;

  void set_sfg(StateFlowGraph *sfg);

  AsyncState get_async_state(SNode *snode, AsyncState::Type type);

  AsyncState get_async_state(Kernel *kernel);

 private:
  StateFlowGraph *sfg_;
  std::unordered_map<IRNode *, uint64> hash_bank_;
  std::unordered_map<IRHandle, std::unique_ptr<IRNode>> ir_bank_;
  std::vector<std::unique_ptr<IRNode>> trash_bin_;  // prevent IR from deleted
  std::unordered_map<std::pair<IRHandle, IRHandle>, IRHandle> fuse_bank_;
  std::unordered_map<IRHandle, IRHandle> demote_activation_bank_;

  // For DSE optimization, the input key is (IRHandle, [SNode*]). This is
  // because it is possible that the same IRHandle may have different sets of
  // SNode stores that are eliminable.
  struct OptimizeDseKey {
    IRHandle task_ir;
    // Intentionally use (ordered) set so that hash is deterministic.
    std::set<const SNode *> eliminable_snodes;

    OptimizeDseKey(const IRHandle task_ir,
                   const std::set<const SNode *> &snodes)
        : task_ir(task_ir), eliminable_snodes(snodes) {
    }

    bool operator==(const OptimizeDseKey &other) const {
      return (task_ir == other.task_ir) &&
             (eliminable_snodes == other.eliminable_snodes);
    }

    bool operator!=(const OptimizeDseKey &other) const {
      return !(*this == other);
    }

    struct Hash {
      std::size_t operator()(const OptimizeDseKey &k) const {
        std::size_t ret = k.task_ir.hash();
        for (const auto *s : k.eliminable_snodes) {
          ret = ret * 100000007UL + reinterpret_cast<uintptr_t>(s);
        }
        return ret;
      }
    };
  };

  std::unordered_map<OptimizeDseKey, IRHandle, OptimizeDseKey::Hash>
      optimize_dse_bank_;
  std::unordered_map<std::size_t, std::size_t> async_state_to_unique_id_;

  std::size_t lookup_async_state_id(void *ptr, AsyncState::Type type);
};

TLANG_NAMESPACE_END
