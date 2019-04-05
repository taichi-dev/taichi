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

Matrix Matrix::identity(int dim) {
  Matrix mat(dim, dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      mat(i, j) = Expr(i == j ? 1.0f : 0.0f);
    }
  }
  return mat;
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


Matrix transposed(const Matrix &A) {
  TC_ASSERT(A.m == A.n);
  Matrix ret(A.m, A.n);
  for (int i = 0; i < A.m; i++) {
    for (int j = 0; j < A.n; j++) {
      ret(i, j).set(A(j, i));
    }
  }
  return ret;
}

TLANG_NAMESPACE_END
