#include "taichi/backends/dx/dx_program.h"

namespace taichi {
namespace lang {

FunctionType DxProgramImpl::compile(Kernel *kernel, OffloadedStmt *offloaded) {
  TI_NOT_IMPLEMENTED
}

void DxProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                        KernelProfilerBase *profiler,
                                        uint64 **result_buffer_ptr) {
  TI_NOT_IMPLEMENTED;
}

void DxProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    std::unordered_map<int, SNode *> &snodes,
    uint64 *result_buffer) {
  TI_NOT_IMPLEMENTED;
}

DxProgramImpl::~DxProgramImpl() {
}

}  // namespace lang
}  // namespace taichi