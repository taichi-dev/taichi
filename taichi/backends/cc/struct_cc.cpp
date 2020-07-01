#include "struct_cc.h"
#include "cc_layout.h"


TLANG_NAMESPACE_BEGIN
namespace cccp {

std::unique_ptr<CCLayout> CCLayoutGen::compile() {
  auto lay = std::make_unique<CCLayout>();
  // W.I.P.
  return lay;
}

}  // namespace cccp
TLANG_NAMESPACE_END
