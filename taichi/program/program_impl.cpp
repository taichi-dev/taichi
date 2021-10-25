#include "program_impl.h"

namespace taichi {
namespace lang {

ProgramImpl::ProgramImpl(CompileConfig &config_) : config(&config_) {
}

void ProgramImpl::compile_snode_tree_types(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees) {
  // FIXME: Eventually all the backends should implement this
  TI_NOT_IMPLEMENTED;
}

}  // namespace lang
}  // namespace taichi
