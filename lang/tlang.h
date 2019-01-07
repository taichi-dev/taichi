#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>
#include <dlfcn.h>

#include "util.h"
#include "visitor.h"
#include "expr.h"
#include "slp_vectorizer.h"
#include "math.h"
#include "codegen_cpu.h"
#include "program.h"
#include "../headers/common.h"

TLANG_NAMESPACE_BEGIN

inline void layout(const std::function<void()> &body) {
  get_current_program().layout(body);
}

inline Kernel kernel(Expr expr, const std::function<void()> &body) {
  return get_current_program().kernel(expr, body);
}

inline Kernel kernel(SNode *snode, const std::function<void()> &body) {
  return get_current_program().kernel(snode, body);
}

inline void group(int n) {
  get_current_program().get_current_kernel().output_group_size = n;
}

inline void parallel_instances(int n) {
  get_current_program().get_current_kernel().parallel_instances = n;
}

inline Adapter &adapter(int i) {
  return get_current_program().get_current_kernel().adapter(i);
}

inline void touch(SNode *snode, Expr target_index, Expr value) {
  auto e = Expr::create(NodeType::touch, target_index, value);
  e->snode_ptr(0) = snode;
  auto &ker = get_current_program().get_current_kernel();
  ker.has_touch = true;
  return ker.ret->ch.push_back(e);
}

TLANG_NAMESPACE_END

/*
 Expr should be what the users play with.
   Simply a ref-counted pointer to nodes, with some operator overloading for
 users to program Node is the IR node, with computational graph connectivity,
 imm, op type etc.

 No double support this time.
 */
