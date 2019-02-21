#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
namespace taichi {
namespace math {
inline int maximum(int a) {
  return a;
}
}  // namespace math
}  // namespace taichi
#include <taichi/math.h>
#include <set>
#include <dlfcn.h>

#include "util.h"
#include "visitor.h"
#include "math.h"
#include "codegen_cpu.h"
#include "program.h"
#include "../headers/common.h"

TLANG_NAMESPACE_BEGIN

inline void layout(const std::function<void()> &body) {
  get_current_program().layout(body);
}

inline Kernel kernel(const std::function<void()> &body) {
  return get_current_program().kernel(body);
}

inline void kernel_name(std::string name) {
  get_current_program().get_current_kernel().name = name;
}

inline void group(int n) {
  get_current_program().get_current_kernel().output_group_size = n;
}

inline void parallel_instances(int n) {
  get_current_program().get_current_kernel().parallel_instances = n;
}

/*
inline void touch(SNode *snode, Expr target_index, Expr value) {
  auto e = Expr::create(NodeType::touch, Expr::load_if_pointer(target_index),
                        Expr::load_if_pointer(value));
  e->snode_ptr(0) = snode;
  auto &ker = get_current_program().get_current_kernel();
  ker.has_touch = true;
  return ker.ret->ch.push_back(e);
}

inline void touch(Expr &expr, Expr target_index, Expr value) {
  return taichi::Tlang::touch(expr->snode_ptr(0)->parent, target_index, value);
}

inline void reduce(Expr target, Expr value) {
  TC_ASSERT(target->type == NodeType::pointer);
  auto e = Expr::create(NodeType::reduce, target, Expr::load_if_pointer(value));
  auto &ker = get_current_program().get_current_kernel();
  return ker.ret->ch.push_back(e);
}
*/

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
void write_partio(std::vector<Vector3> positions, const std::string &file_name);
TC_NAMESPACE_END

/*
 Expr should be what the users play with.
   Simply a ref-counted pointer to nodes, with some operator overloading for
 users to program Node is the IR node, with computational graph connectivity,
 imm, op type etc.

 No double support this time.
 */
