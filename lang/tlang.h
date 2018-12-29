#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>
#include <dlfcn.h>

#include "util.h"
#include "visitor.h"
#include "expr.h"
#include "address.h"
#include "memory_allocator.h"
#include "vectorizer.h"
#include "math.h"
#include "codegen.h"
#include "program.h"
#include "../headers/common.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

inline std::pair<int64, int64> range(int64 start, int64 end) {
  return {start, end};
}

using ForBody = std::function<void()>;
inline void for_loop(Index &index,
                     std::pair<int64, int64> r,
                     const ForBody &body) {
  auto &prog = get_current_program();
  TC_ASSERT(r.first == 0);
  get_current_program().current_function->n = r.second;
  body();
}
}  // namespace Tlang

TC_NAMESPACE_END

/*
 Expr should be what the users play with.
   Simply a ref-counted pointer to nodes, with some operator overloading for
 users to program Node is the IR node, with computational graph connectivity,
 imm, op type etc.

 No double support this time.
 */
