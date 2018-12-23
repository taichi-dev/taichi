#pragma once

#include "tlang.h"

namespace taichi::Tlang {

using Real = Expr;

template <typename T>
using Var = Expr;

using Float32 = Var<float32>;
using Float = Float32;
using Int32 = Var<int32>;
using Int = Int32;

struct Matrix {
  using T = Float;
  int n, m;
  std::vector<T> entries;

  Matrix() {
    n = m = 0;
  }

  bool initialized() {
    return n * m >= 1;
  }

  Matrix(int n, int m = 1) : n(n), m(m) {
    TC_ASSERT(n * m >= 1);
    entries.resize(n * m, T());
  }

  T &operator()(int i, int j) {
    TC_ASSERT(0 <= i && i < n);
    TC_ASSERT(0 <= j && j < n);
    return entries[i * m + j];
  }

  const T &operator()(int i, int j) const {
    TC_ASSERT(0 <= i && i < n);
    TC_ASSERT(0 <= j && j < n);
    return entries[i * m + j];
  }

  T &operator()(int i) {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  const T &operator()(int i) const {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  Matrix &operator=(const Matrix &o) {
    if (initialized()) {
      n = o.n;
      m = o.m;
      entries = o.entries;
    } else {
      TC_ASSERT(n == o.n && m == o.m);
      for (int i = 0; i < (int)entries.size(); i++) {
        entries[i] = o.entries[i];
      }
    }
    return *this;
  }

  Matrix element_wise_prod(const Matrix &o) {
    TC_ASSERT(n == o.n && m == o.m);
    Matrix ret(n, m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        ret(i, j) = (*this)(i, j) + o(i, j);
      }
    }
    return ret;
  }

  Matrix operator[](Expr index) {
    Matrix ret(n, m);
    for (int i = 0; i < n * m; i++) {
      ret.entries[i] = entries[i][index];
    }
    return ret;
  }
};

inline Matrix operator*(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.m == B.n);
  Matrix C(A.n, B.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A(i, 0) * B(0, j);
      for (int k = 1; k < A.m; k++) {
        C(i, j) = C(i, j) + A(i, k) * B(k, j);
      }
    }
  }
  return C;
}

inline Matrix operator+(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.n == B.n);
  TC_ASSERT(A.m == B.m);
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
  return C;
}

inline Matrix operator-(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.n == B.n);
  TC_ASSERT(A.m == B.m);
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = A(i, j) - B(i, j);
    }
  }
  return C;
}

using Vector = Matrix;

inline Expr floor(const Expr &a) {
  return Expr::create(NodeType::floor, Expr::load_if_pointer(a));
}

inline Expr max(const Expr &a, const Expr &b) {
  auto n = Expr::create(NodeType::max, a, b);
  n->data_type = a->data_type;
  return n;
}

inline Expr min(const Expr &a, const Expr &b) {
  auto n = Expr::create(NodeType::min, a, b);
  n->data_type = a->data_type;
  return n;
}

inline Expr imm(int i) {
  auto n = Expr::create(NodeType::imm);
  n->value<int32>() = i;
  n->data_type = DataType::i32;
  return n;
}

inline Expr imm(float32 i) {
  auto n = Expr::create(NodeType::imm);
  n->data_type = DataType::f32;
  n->value<float32>() = i;
  return n;
}

template <typename T>
inline Expr cast(const Expr &i) {
  auto n = Expr::create(NodeType::cast, i);
  if (std::is_same<T, int32>()) {
    n->data_type = DataType::i32;
  } else {
    n->data_type = DataType::f32;
  }
  return n;
}

inline Float32 lerp(Float a, Float x0, Float x1) {
  return (imm(1.0_f) - a) * x0 + a * x1;
}
}