#include "tlang.h"

TLANG_NAMESPACE_BEGIN

void layout(const std::function<void()> &body) {
  get_current_program().layout(body);
}

TLANG_NAMESPACE_END
