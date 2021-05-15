#include "taichi/struct/struct.h"

namespace taichi {
namespace lang {

class FakeStructCompiler : public StructCompiler {
 public:
  FakeStructCompiler() : StructCompiler(/*prog=*/nullptr) {
  }

  void generate_types(SNode &) override {
  }

  void generate_child_accessors(SNode &) override {
  }

  void run(SNode &root, bool host) override {
    infer_snode_properties(root);
    // TODO(#2327): Stop calling this
    compute_trailing_bits(root);
  }
};

}  // namespace lang
}  // namespace taichi
