#include <taichi/taichi>
#include <set>
#include "../ir.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {

class Offloader {
 public:
  Offloader(IRNode *root) {
    run(root);
  }

  void run(IRNode *root) {
    auto root_block = dynamic_cast<Block *>(root);
    auto &root_statements = root_block->statements;
    for (int i = 0; i < (int)root_statements.size(); i++) {

    }
  }
};

void offload(IRNode *root) {
  Offloader _(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
