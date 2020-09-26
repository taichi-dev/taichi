#include "taichi/program/async_utils.h"

TLANG_NAMESPACE_BEGIN

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

  std::unordered_map<IRHandle, TaskMeta> meta_bank_;
  std::unordered_map<IRHandle, TaskFusionMeta> fusion_meta_bank_;

 private:
  std::unordered_map<IRNode *, uint64> hash_bank_;
  std::unordered_map<IRHandle, std::unique_ptr<IRNode>> ir_bank_;
  std::vector<std::unique_ptr<IRNode>> trash_bin;  // prevent IR from deleted
  std::unordered_map<std::pair<IRHandle, IRHandle>, IRHandle> fuse_bank_;
  std::unordered_map<IRHandle, IRHandle> demote_activation_bank_;
};

TLANG_NAMESPACE_END
