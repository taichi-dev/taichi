
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

  void run(SNode &, bool) override {
  }
};

}  // namespace lang
}  // namespace taichi
