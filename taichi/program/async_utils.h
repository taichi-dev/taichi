#pragma once

#include <unordered_map>
#include <unordered_set>

#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

struct TaskMeta;

class IRHandle {
 public:
  IRHandle() : ir_(nullptr), hash_(0) {
  }

  IRHandle(const IRNode *ir, uint64 hash) : ir_(ir), hash_(hash) {
  }

  std::unique_ptr<IRNode> clone() const;

  const IRNode *ir() const {
    return ir_;
  }

  uint64 hash() const {
    return hash_;
  }

  bool empty() const {
    return ir_ == nullptr;
  }

  // Two IRHandles are considered the same iff their hash values are the same.
  bool operator==(const IRHandle &other_ir_handle) const {
    return hash_ == other_ir_handle.hash_;
  }

 private:
  const IRNode *ir_;  // not owned
  uint64 hash_;
};

TLANG_NAMESPACE_END

namespace std {
template <>
struct hash<taichi::lang::IRHandle> {
  std::size_t operator()(const taichi::lang::IRHandle &ir_handle) const
      noexcept {
    return ir_handle.hash();
  }
};

template <>
struct hash<std::pair<taichi::lang::IRHandle, taichi::lang::IRHandle>> {
  std::size_t operator()(
      const std::pair<taichi::lang::IRHandle, taichi::lang::IRHandle>
          &ir_handles) const noexcept {
    return ir_handles.first.hash() * 100000007UL + ir_handles.second.hash();
  }
};
}  // namespace std

TLANG_NAMESPACE_BEGIN

// Records the necessary data for launching an offloaded task.
class TaskLaunchRecord {
 public:
  Context context;
  Kernel *kernel;  // TODO: remove this
  IRHandle ir_handle;

  TaskLaunchRecord();

  TaskLaunchRecord(Context context, Kernel *kernel, IRHandle ir_handle);

  OffloadedStmt *stmt() const;

  bool empty() const;
};

struct AsyncState {
  enum class Type { mask, value, list };

  AsyncState(SNode *snode, Type type) : snode(snode), type(type) {
  }

  SNode *snode;
  Type type;

  bool operator<(const AsyncState &other) const {
    return snode < other.snode || (snode == other.snode && type < other.type);
  }

  bool operator==(const AsyncState &other) const {
    return snode == other.snode && type == other.type;
  }

  std::string name() const {
    std::string type_name;
    switch (type) {
      case Type::mask:
        type_name = "mask";
        break;
      case Type::value:
        type_name = "value";
        break;
      case Type::list:
        type_name = "list";
        break;
    }
    return snode->get_node_type_name_hinted() + "_" + type_name;
  }
};

class AsyncStateHash {
 public:
  size_t operator()(const AsyncState &s) const {
    return (uint64)s.snode ^ (uint64)s.type;
  }
};

struct TaskMeta {
  std::string name;
  OffloadedStmt::TaskType type{OffloadedStmt::TaskType::serial};
  SNode *snode{nullptr};  // struct-for and listgen only
  std::unordered_set<AsyncState, AsyncStateHash> input_states;
  std::unordered_set<AsyncState, AsyncStateHash> output_states;

  void print() const;
};

class IRBank;

TaskMeta *get_task_meta(IRBank *bank, const TaskLaunchRecord &t);

TLANG_NAMESPACE_END
