// Codegen for the hierarchical data structure
#pragma once

#include <taichi/platform/metal/metal_data_types.h>
#include <taichi/platform/metal/metal_kernel_util.h>
#include <taichi/snode.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

TLANG_NAMESPACE_BEGIN
namespace metal {

class MetalStructCompiler {
 public:
  using CompiledResult = metal::StructCompiledResult;

  CompiledResult run(SNode &node);

 private:
  void collect_snodes(SNode &snode);
  void generate_types(const SNode &snode);
  size_t compute_snode_size(const SNode &sn);

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    src_code_ += fmt::format(f, std::forward<Args>(args)...) + '\n';
  }

  std::vector<SNode *> snodes_;
  std::string src_code_;
};

}  // namespace metal
TLANG_NAMESPACE_END
