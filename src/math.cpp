#include "math.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

void Matrix::fill_global(DataType dt) {
  for (int i = 0; i < n * m; i++) {
    entries[i].set(global_new(dt));
  }
}

Expr &Matrix::operator()(int i, int j) {
  TC_ASSERT(0 <= i && i < n);
  TC_ASSERT(0 <= j && j < m);
  return entries[i * m + j];
}

const Expr &Matrix::operator()(int i, int j) const {
  TC_ASSERT(0 <= i && i < n);
  TC_ASSERT(0 <= j && j < m);
  return entries[i * m + j];
}

void Mutable(Matrix &mat, DataType dt) {
  for (int i = 0; i < mat.entries.size(); i++) {
    declare_unnamed_var(mat.entries[i], dt);
  }
}

SNode &SNode::place(Matrix &mat) {
  for (auto &e : mat.entries) {
    this->place(e);
  }
  return *this;
}

TLANG_NAMESPACE_END
