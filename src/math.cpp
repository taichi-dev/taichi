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

SNode &SNode::place(Matrix &mat) {
  for (auto &e : mat.entries) {
    this->place(e);
  }
  return *this;
}

Matrix transposed(const Matrix &A) {
  TC_ASSERT(A.m == A.n);
  Matrix ret(A.m, A.n);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      ret(i, j).set(A(j, i));
    }
  }
  return ret;
}

Matrix diag_matrix(const Matrix &A) {
  TC_ASSERT(A.m == 1);
  Matrix ret(A.n, A.n);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.n; j++) {
      if (i == j) {
        ret(i, j).set(A(i, 0));
      } else {
        ret(i, j) = Expr(0.0_f);
      }
    }
  }
  return ret;
}

Matrix &&Var(Matrix &&mat) {
  for (int i = 0; i < mat.entries.size(); i++) {
    declare_unnamed_var(mat.entries[i], DataType::unknown);
  }
  return std::move(mat);
}

Matrix Var(const Matrix &mat_) {
  Matrix mat;
  mat.set(mat_);
  for (int i = 0; i < mat.entries.size(); i++) {
    declare_unnamed_var(mat.entries[i], DataType::unknown);
  }
  return mat;
}

TLANG_NAMESPACE_END
